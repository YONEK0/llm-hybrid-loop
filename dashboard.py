"""dashboard.py — MiMo-style live training dashboard for the V5 L0 run (v2 rewrite).

Independent, read-only process: never touches CUDA or writes under runs/. Watches
runs/<run>/history.jsonl (byte-offset tail, torn-line tolerant) plus watch.log and
the latest checkpoint metadata. Serves a dark single-page board in the spirit of
MiMo's open training stream, adapted to our metrics:

  status bar      RUNNING / PAUSED / COMPLETED chip + phase + server time
  stat cards      step, supervised tokens %, speed (s/batch + sup tok/s), ETA,
                  depth gain (d1-d4), A_bar, q_hat, max grad norm
  HERO chart      dev CE (d1/d2/d4[/d8]) vs supervised tokens, baseline dashed
  acceptance      CE-vs-depth chart + S1b frozen-reference chart (own scale)
  proprietary     block-hidden-norm vs depth (S1b mechanism), A_bar series,
                  q_hat (confidence head) series, per-supervision-depth lm bars
  throughput      coarse tok/s from watch.log (fine per-batch series after resume,
                  trainer now stamps each batch with "ts")
  feeds           training events (start/baseline/eval/final) + acceptance verdicts
  samples         greedy generations from the latest eval monitor

Usage:  loopus_env/Scripts/python.exe dashboard.py [--port 8741] [--host 0.0.0.0] [--run v5_l0]
"""

import argparse
import json
import re
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent


class Tail:
    """Incremental, tear-tolerant reader for the append-only history file."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.records = []      # per-batch records (have 'logs')
        self.evals = []        # eval events
        self.events = []       # all event records (start/baseline/eval/final)
        self.baseline = None
        self.start_args = None
        self.load_errors = 0

    def poll(self):
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:
            self.offset, self.records, self.evals, self.events = 0, [], [], []
        if size == self.offset:
            return
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return
        self.offset += cut + 1
        for raw in chunk[:cut].split(b"\n"):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                self.load_errors += 1
                continue
            ev = rec.get("event")
            if ev:
                self.events.append(rec)
                if ev == "eval":
                    self.evals.append(rec)
                elif ev == "dev_baseline":
                    self.baseline = rec.get("dev_ce")
                elif ev == "start":
                    self.start_args = rec.get("args")
            elif "logs" in rec:
                self.records.append(rec)

    # ---------- views ----------
    def status(self):
        if any(r.get("event") == "final" for r in self.events):
            return "COMPLETED"
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return "UNKNOWN"
        return "RUNNING" if age < 300 else "PAUSED"

    def progress(self):
        if not self.records:
            return {"batches": 0}
        last = self.records[-1]
        args = self.start_args or {}
        target = float(args.get("target_supervised_tokens", 5e6))
        sup = last["sup_tokens"]
        k = int(args.get("K", 5))
        ctx = int(args.get("ctx", 512))
        spb = self.evals[-1].get("sec_per_batch") if self.evals else None
        remaining = max(target - sup, 0) / (k * ctx)
        gain = None
        if self.evals and self.baseline:
            ce = self.evals[-1]["dev_ce"]
            if isinstance(ce.get("d1"), (int, float)) and isinstance(ce.get("d4"), (int, float)):
                base_gain = self.baseline.get("d1", 0) - self.baseline.get("d4", 0)
                gain = {"d1_minus_d4": round(ce["d1"] - ce["d4"], 4),
                        "vs_baseline_gain": round((ce["d1"] - ce["d4"]) - base_gain, 4)}
        # batches = true step count (unique); records = raw log lines, which include
        # the re-run tail after a hard stop resumes from the last checkpoint
        return {"batches": last["step"], "records": len(self.records),
                "step": last["step"],
                "sup_tokens": sup, "target_tokens": target,
                "fraction": round(sup / target, 5),
                "sec_per_batch": spb,
                "eta_hours": (round(remaining * spb / 3600, 2) if spb else None),
                "B": args.get("B"), "K": k, "ctx": ctx,
                "depth_gain": gain, "phase": "L0 · gate+head only · M=L8-19"}

    def depth_curves(self):
        out = {"baseline": self.baseline, "evals": []}
        for e in self.evals:
            out["evals"].append({"step": e["step"], "sup_tokens": e["sup_tokens"],
                                 "ce": e["dev_ce"]})
        try:
            s1b = json.loads((ROOT / "results/s1b_trace_v2.json")
                             .read_text(encoding="utf-8"))
            ce = s1b["models"]["qwen3.5-4b_hybrid_nf4"]["candidates"]["L8-19"]["mean"]["ce"]
            out["frozen_L8_19"] = {str(b): v for b, v in enumerate(ce)}
        except Exception:
            out["frozen_L8_19"] = None
        return out

    def gate(self):
        rec = self.records[-150:]
        abar = [l.get("A_bar_mean") for r in rec for l in r["logs"]
                if l.get("A_bar_mean") is not None]
        qhat = [l.get("q_hat") for r in rec for l in r["logs"]
                if l.get("q_hat") is not None]
        gn = [l.get("grad_norm") for r in rec for l in r["logs"]
              if l.get("grad_norm") is not None]
        by_b = {}
        for r in rec:
            for l in r["logs"]:
                by_b.setdefault(l["b"], []).append(l["lm"])
        return {"A_bar_mean_recent": round(sum(abar) / len(abar), 5) if abar else None,
                "A_bar_series": [round(v, 4) for v in abar[-120:]],
                "q_hat_series": [round(v, 4) for v in qhat[-120:]],
                "q_hat_mean": round(sum(qhat) / len(qhat), 4) if qhat else None,
                "per_b_lm_recent": {str(b): round(sum(v) / len(v), 4)
                                    for b, v in sorted(by_b.items())},
                "grad_norm_recent_max": round(max(gn), 3) if gn else None}

    def norm_series(self):
        out = []
        for e in self.evals:
            mon = e.get("dev_ce", {}).get("monitor", {})
            if mon.get("block_hidden_norm"):
                out.append({"step": e["step"], "norm": mon["block_hidden_norm"]})
        return out

    def samples(self):
        for e in reversed(self.evals):
            mon = e.get("dev_ce", {}).get("monitor", {})
            if mon.get("samples"):
                return {"step": e["step"], "samples": mon["samples"]}
        return None

    def throughput(self, run_dir):
        spb = self.evals[-1].get("sec_per_batch") if self.evals else None
        sup_rate = None
        if spb and self.start_args:
            sup_rate = round(int(self.start_args.get("K", 5))
                             * int(self.start_args.get("ctx", 512)) / spb, 1)
        # fine-grained: per-batch throughput from history timestamps (records since
        # the trainer started stamping "ts"); smoothed with a 15-batch window
        fine = []
        pts = [(r["ts"], r["sup_tokens"]) for r in self.records
               if isinstance(r.get("ts"), (int, float))]
        for (t1, s1), (t2, s2) in zip(pts, pts[1:]):
            dt = t2 - t1
            if dt > 0:
                rate = (s2 - s1) / dt
                if 0 < rate < 500:          # sanity: exclude resume-gap artifacts
                    fine.append({"step": None, "tok_s": round(rate, 1)})
        # smooth
        for i in range(len(fine)):
            w = fine[max(0, i - 14):i + 1]
            fine[i]["tok_s"] = round(sum(p["tok_s"] for p in w) / len(w), 1)
        fine = fine[-120:]
        coarse = []
        wl = run_dir / "watch.log"
        if wl.exists():
            pts2 = []
            for ln in wl.read_text(encoding="utf-8").splitlines():
                m = re.match(r"(\S+ \S+) verdict=(\w+) step=\d+ evals=\d+ sup=(\d+)", ln)
                if m:
                    pts2.append((m.group(1), m.group(2), int(m.group(3))))
            for (t1, _, s1), (t2, v2, s2) in zip(pts2, pts2[1:]):
                try:
                    dt = time.mktime(time.strptime(t2, "%Y-%m-%d %H:%M:%S")) \
                        - time.mktime(time.strptime(t1, "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
                if dt > 0 and s2 > s1:
                    rate = (s2 - s1) / dt
                    if rate < 500:          # drop pause-spanning artifacts
                        coarse.append({"t": t2.split()[1][:5], "verdict": v2,
                                       "tok_s": round(rate, 1)})
        return {"sec_per_batch": spb, "sup_tok_s": sup_rate,
                "fine": fine[-120:], "coarse": coarse[-40:]}

    def badges(self):
        out = []
        base, last = self.baseline, (self.evals[-1] if self.evals else None)
        if base and last:
            ce = last["dev_ce"]
            dd = {k: v for k, v in ce.items()
                  if k.startswith("d") and isinstance(v, (int, float))}
            if dd:
                drop = all(dd[k] < base[k] for k in dd if k in base)
                out.append({"name": "CE 全面低于基线", "ok": drop,
                            "detail": " ".join(f"{k}={v}" for k, v in sorted(dd.items()))})
            if "d1" in dd and "d4" in dd:
                out.append({"name": "深度倒置 d4≤d1", "ok": dd["d4"] <= dd["d1"],
                            "detail": f"d4−d1={round(dd['d4'] - dd['d1'], 3)}"})
        g = self.gate()
        if g["A_bar_mean_recent"] is not None:
            a = g["A_bar_mean_recent"]
            out.append({"name": "门开度正常", "ok": 0.001 < a < 0.9, "detail": f"A_bar={a}"})
        if g["grad_norm_recent_max"] is not None:
            gn = g["grad_norm_recent_max"]
            out.append({"name": "梯度平稳", "ok": gn < 50, "detail": f"max‖g‖={gn}"})
        out.append({"name": "误差计数", "ok": self.load_errors == 0,
                    "detail": f"解析失败 {self.load_errors} 行"})
        return out

    def feeds(self, run_dir):
        train = []
        for r in self.events:
            ev = r.get("event")
            if ev == "start":
                a = r.get("args", {})
                train.append(f"[start] B={a.get('B')} K={a.get('K')} ctx={a.get('ctx')} "
                             f"目标 {float(a.get('target_supervised_tokens', 0)) / 1e6:.0f}M "
                             f"ckpt-every={a.get('ckpt_every')}")
            elif ev == "dev_baseline":
                c = {k: v for k, v in r.get("dev_ce", {}).items()
                     if k.startswith("d") and isinstance(v, (int, float))}
                train.append("[基线] " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
            elif ev == "eval":
                c = {k: v for k, v in r.get("dev_ce", {}).items()
                     if k.startswith("d") and isinstance(v, (int, float))}
                mon = r.get("dev_ce", {}).get("monitor", {})
                extra = ""
                if mon.get("block_hidden_norm"):
                    extra += " ‖h‖ " + ",".join(f"{k}:{v}" for k, v
                                                in mon["block_hidden_norm"].items())
                if mon.get("samples"):
                    extra += f" 样本×{len(mon['samples'])}"
                train.append(f"[eval s{r['step']}] "
                             + " ".join(f"{k}={v}" for k, v in sorted(c.items())) + extra)
            elif ev == "final":
                train.append(f"[final] step {r.get('step')} 结束")
        acc = []
        wl = run_dir / "watch.log"
        if wl.exists():
            acc = [l.strip() for l in wl.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
        return {"train": train[-30:][::-1], "accept": acc[-30:][::-1]}

    def snapshot(self, run_dir):
        self.poll()
        try:
            ck = run_dir / "ckpt_latest.pt"
            st = ck.stat()
            ckpt = {"size_mb": round(st.st_size / 2 ** 20, 1),
                    "age_sec": round(time.time() - st.st_mtime),
                    "step": (self.records[-1]["step"] if self.records else None)}
        except OSError:
            ckpt = None
        try:
            st = self.path.stat()
            hist = {"age_sec": round(time.time() - st.st_mtime)}
        except OSError:
            hist = None
        return {"status": self.status(), "progress": self.progress(),
                "hero": {"baseline": self.baseline,
                         "series": [{"step": e["step"], "sup": e["sup_tokens"],
                                     "ce": {k: v for k, v in e["dev_ce"].items()
                                            if k.startswith("d")}}
                                    for e in self.evals]},
                "depth_curves": self.depth_curves(),
                "gate": self.gate(),
                "norm_series": self.norm_series(),
                "samples": self.samples(),
                "throughput": self.throughput(run_dir),
                "badges": self.badges(),
                "feeds": self.feeds(run_dir),
                "checkpoint": ckpt, "history_file": hist,
                "parse_errors": self.load_errors,
                "server_time": time.strftime("%H:%M:%S")}


HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>V5 L0 · 训练监控</title><style>
body{font-family:Consolas,'Microsoft YaHei',monospace;background:#0d1117;color:#d8dee6;margin:0;padding:14px}
h1{font-size:17px;color:#9cd32e;margin:0 0 4px}
.sub{color:#6b7684;font-size:12px;margin-bottom:10px}
.chip{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:bold;vertical-align:middle}
.RUNNING{background:#1d3b1d;color:#6ee76e}.PAUSED{background:#3b331d;color:#ffd35e}
.COMPLETED{background:#1d2b3b;color:#6ec3ff}.UNKNOWN{background:#333;color:#aaa}
.grid{display:grid;gap:10px}
.stats{grid-template-columns:repeat(4,1fr)}
.card{background:#161b22;border:1px solid #2a313c;border-radius:8px;padding:10px 14px}
.stat .v{font-size:20px;color:#e8edf3}.stat .k{font-size:11px;color:#6b7684}
.stat .d{font-size:11px;color:#8b949e}
.row2{grid-template-columns:1fr 1fr}.row3{grid-template-columns:1fr 1fr 1fr}
.badge{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #21262d;font-size:12px}
.ok{color:#6ee76e}.bad{color:#ff6b6b}.warn{color:#ffd35e}
.feed{max-height:220px;overflow-y:auto;font-size:11.5px;line-height:1.55;color:#9aa4af}
.feed .hl{color:#d8dee6}
.samp{background:#0d1117;border:1px solid #2a313c;border-radius:6px;padding:6px 8px;margin:4px 0;font-size:12px;color:#c9d1d9}
h2{font-size:13px;color:#79b8ff;margin:0 0 6px}
svg{background:#0d1117;border:1px solid #22272e;width:100%;height:auto}
.legend span{margin-right:12px;font-size:11.5px}
table.top{width:100%;font-size:12px}
td{padding:1px 8px 1px 0}.k{color:#6b7684}.v{color:#e8edf3}
</style></head><body>
<h1>V5 L0 训练监控 — Qwen3.5-4B hybrid 循环化 <span id="chip" class="chip">…</span></h1>
<div class="sub" id="subinfo"></div>
<div id="paused_note" style="display:none;background:#3b331d;color:#ffd35e;border:1px solid #6b5a1f;border-radius:6px;padding:6px 12px;margin-bottom:10px;font-size:12.5px"></div>

<div class="grid stats" id="stats"></div>

<div class="grid row2" style="margin-top:10px">
 <div class="card"><h2>主曲线 · dev CE 随监督 token（验收信号：三条线整体下移且 d4 压到 d1 之下）</h2>
  <div class="legend"><span style="color:#7cf">d1</span><span style="color:#5f5">d2</span>
  <span style="color:#fd5">d4</span><span style="color:#f5f">d8</span>
  <span style="color:#888">- - 训练前基线</span></div>
  <svg id="ch_hero"></svg></div>
 <div class="card"><h2>验收徽章 / 模型实时输出（最新评估点贪心样本）</h2>
  <div id="badges"></div>
  <div style="margin-top:8px"><span class="k" style="font-size:11px">SAMPLES</span>
  <div id="samples"></div></div></div>
</div>

<div class="grid row3" style="margin-top:10px">
 <div class="card"><h2>CE(d) 曲线族 · 基线 vs 评估点</h2>
  <div class="legend"><span style="color:#888">- - 基线</span><span style="color:#7cf">■ 评估点</span></div>
  <svg id="ch_depth"></svg></div>
 <div class="card"><h2>S1b 冻结参照（无训练会走向 8.3）vs 训练点</h2>
  <div class="legend"><span style="color:#fb6">- - 冻结循环</span><span style="color:#7cf">● 训练后</span></div>
  <svg id="ch_frozen"></svg></div>
 <div class="card"><h2>专属 · 块后隐状态范数 vs 深度（S1b 机制：门应压住膨胀）</h2>
  <div class="legend"><span style="color:#7cf">■ 各评估点</span></div>
  <svg id="ch_norm"></svg></div>
</div>

<div class="grid row3" style="margin-top:10px">
 <div class="card"><h2>门开度 A_bar（近期监督步）</h2><svg id="ch_gate"></svg></div>
 <div class="card"><h2>置信头 q_hat（近期监督步）</h2><svg id="ch_q"></svg></div>
 <div class="card"><h2>吞吐 · 监督 token/s（验收间隔粗粒度）</h2><svg id="ch_tp"></svg></div>
</div>

<div class="card" style="margin-top:10px"><h2>按监督深度 b 的近期 lm（b0..19 天然递增，训练应整体下移）</h2>
<svg id="ch_b"></svg></div>

<div class="grid row2" style="margin-top:10px">
 <div class="card"><h2>事件流 · 训练事件</h2><div class="feed" id="feed_train"></div></div>
 <div class="card"><h2>事件流 · 定时验收（watch.log）</h2><div class="feed" id="feed_acc"></div></div>
</div>
<div class="sub" style="margin-top:8px" id="foot"></div>

<script>
const DEPTHS=[1,2,4],DEPTHS8=[1,2,4,8];
function el(t,a){const e=document.createElementNS('http://www.w3.org/2000/svg',t);
 for(const k in (a||{}))e.setAttribute(k,a[k]);return e}
function frame(svg,X,Y,W,H,xmax,ymin,ymax,xl,yl,xt){
 svg.appendChild(el('line',{x1:X,y1:Y+H,x2:X+W,y2:Y+H,stroke:'#333c47'}));
 svg.appendChild(el('line',{x1:X,y1:Y,x2:X,y2:Y+H,stroke:'#333c47'}));
 xt.forEach(t=>{const x=X+W*t/xmax;
  svg.appendChild(el('line',{x1:x,y1:Y,x2:x,y2:Y+H,stroke:'#1d232b'}));
  const tx=el('text',{x:x+3,y:Y+13,fill:'#6b7684','font-size':11});tx.textContent=t;svg.appendChild(tx)});
 const a=el('text',{x:12,y:Y+13,fill:'#6b7684','font-size':11});a.textContent=ymax.toFixed(1);svg.appendChild(a);
 const b=el('text',{x:12,y:Y+H-3,fill:'#6b7684','font-size':11});b.textContent=ymin.toFixed(1);svg.appendChild(b);
 const c=el('text',{x:X+W-56,y:Y+H+20,fill:'#6b7684','font-size':11});c.textContent=xl;svg.appendChild(c);
 const e=el('text',{x:12,y:Y-5,fill:'#6b7684','font-size':11});e.textContent=yl;svg.appendChild(e)}
function line(svg,pts,color,dash,w){let d='';pts.forEach((p,i)=>{d+=(i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)});
 svg.appendChild(el('path',{d:d,fill:'none',stroke:color,'stroke-width':w||1.8,'stroke-dasharray':dash||'none'}))}
function dot(svg,x,y,c,r){svg.appendChild(el('circle',{cx:x,cy:y,r:r||3.5,fill:c,stroke:'#0d1117'}))}
function label(svg,x,y,t,c){const e=el('text',{x:x,y:y,fill:c||'#9aa4af','font-size':11});e.textContent=t;svg.appendChild(e)}
function mkY(vals,padf){let lo=Math.min(...vals),hi=Math.max(...vals);const p=(hi-lo)*padf+0.01;return[lo-p,hi+p]}

function drawHero(d){
 const s=document.getElementById('ch_hero');s.setAttribute('viewBox','0 0 860 340');s.innerHTML='';
 const S=d.hero.series||[];
 if(!S.length){label(s,20,30,'等待首个评估点…','#6b7684');return}
 const has8=S.some(e=>typeof e.ce.d8==='number');
 let vals=[];S.forEach(e=>DEPTHS8.forEach(k=>{if(typeof e.ce['d'+k]==='number')vals.push(e.ce['d'+k])}));
 if(d.hero.baseline)(has8?DEPTHS8:DEPTHS).forEach(k=>{const v=d.hero.baseline['d'+k];
  if(typeof v==='number')vals.push(v)});
 const[ymin,ymax]=mkY(vals,0.18);
 const X=52,Y=18,W=780,H=290,xm=Math.max(...S.map(e=>e.sup/1e6))*1.06;
 frame(s,X,Y,W,H,xm,ymin,ymax,'监督 token (M)','dev CE',[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]);
 const y=v=>Y+H-(v-ymin)/(ymax-ymin)*H,x=v=>X+W*v/xm;
 const C={1:'#7cf',2:'#5f5',4:'#fd5',8:'#f5f'};
 if(d.hero.baseline)DEPTHS8.forEach(k=>{const v=d.hero.baseline['d'+k];
  if(typeof v==='number'){s.appendChild(el('line',{x1:X,y1:y(v),x2:X+W,y2:y(v),stroke:'#555','stroke-dasharray':'5,4'}));
   label(s,X+W+2,y(v)+4,v.toFixed(2),'#777')}});
 const cols=[ '#8fd0ff','#8fe08f','#ffe08f','#f0a8ff'];
 DEPTHS8.forEach((k,ki)=>{const pts=S.filter(e=>typeof e.ce['d'+k]==='number')
  .map(e=>[x(e.sup/1e6),y(e.ce['d'+k])]);
  if(pts.length){line(s,pts,cols[ki]);pts.forEach(p=>dot(s,p[0],p[1],cols[ki],3));
   const last=S.filter(e=>typeof e.ce['d'+k]==='number').pop();
   label(s,pts[pts.length-1][0]+6,pts[pts.length-1][1]+4,'d'+k+'='+last.ce['d'+k].toFixed(3),cols[ki])}});
}
function drawDepth(d){
 const s=document.getElementById('ch_depth');s.setAttribute('viewBox','0 0 430 260');s.innerHTML='';
 const dc=d.depth_curves,vals=[];
 (dc.evals||[]).forEach(e=>DEPTHS.forEach(k=>{if(typeof e.ce['d'+k]==='number')vals.push(e.ce['d'+k])}));
 if(dc.baseline)DEPTHS.forEach(k=>{if(typeof dc.baseline['d'+k]==='number')vals.push(dc.baseline['d'+k])});
 if(!vals.length){label(s,20,30,'数据积累中…','#6b7684');return}
 const[ymin,ymax]=mkY(vals,0.15);const X=44,Y=16,W=370,H=210,xm=4.6;
 frame(s,X,Y,W,H,xm,ymin,ymax,'d','CE',[1,2,3,4]);
 const y=v=>Y+H-(v-ymin)/(ymax-ymin)*H,x=k=>X+W*k/xm;
 if(dc.baseline)line(s,DEPTHS.map(k=>[x(k),y(dc.baseline['d'+k])]),'#888','5,3',1.4);
 (dc.evals||[]).forEach((e,i)=>{const c=['#7cf','#5f5','#fd5','#f5f','#5ff'][i%5];
  line(s,DEPTHS.map(k=>[x(k),y(e.ce['d'+k])]),c);
  DEPTHS.forEach(k=>dot(s,x(k),y(e.ce['d'+k]),c,3))});
}
function drawFrozen(d){
 const s=document.getElementById('ch_frozen');s.setAttribute('viewBox','0 0 430 260');s.innerHTML='';
 const fr=d.depth_curves.frozen_L8_19;
 if(!fr){label(s,20,30,'无参照数据','#6b7684');return}
 const fv=Object.values(fr);let lo=Math.min(...fv),hi=Math.max(...fv);const p=(hi-lo)*0.08;
 const X=44,Y=16,W=370,H=210;
 frame(s,X,Y,W,H,20,lo-p,hi+p,'d（0=跳过循环块）','CE',[0,4,8,12,16,20]);
 const y=v=>Y+H-(v-lo-p)/(hi+p-lo-p)*H,x=k=>X+W*k/20;
 line(s,Object.entries(fr).map(([k,v])=>[x(+k),y(v)]),'#fb6','5,4',1.6);
 const last=(d.depth_curves.evals||[]).slice(-1)[0];
 if(last)DEPTHS.forEach(k=>{if(typeof last.ce['d'+k]==='number'){
  dot(s,x(k),y(last.ce['d'+k]),'#7cf',5);
  label(s,x(k)+7,y(last.ce['d'+k])+4,'d'+k+'='+last.ce['d'+k].toFixed(2),'#9cd32e')}});
 label(s,X+W-90,Y+14,'冻结 b20='+fr['20'].toFixed(2),'#fb6');
}
function drawNorm(d){
 const s=document.getElementById('ch_norm');s.setAttribute('viewBox','0 0 430 260');s.innerHTML='';
 const ns=d.norm_series||[];
 if(!ns.length){label(s,20,30,'续跑后首个评估点起记录…','#6b7684');return}
 const allv=[];ns.forEach(e=>Object.values(e.norm).forEach(v=>allv.push(v)));
 const[ymin,ymax]=mkY(allv,0.12);const X=44,Y=16,W=370,H=210,xm=8.6;
 frame(s,X,Y,W,H,xm,ymin,ymax,'d','‖h‖ L2',[1,2,4,8]);
 const y=v=>Y+H-(v-ymin)/(ymax-ymin)*H,x=k=>X+W*k/xm;
 const cols=['#7cf','#5f5','#fd5','#f5f','#5ff'];
 ns.forEach((e,i)=>{const c=cols[i%5],hn=e.norm;
  const ks=Object.keys(hn).map(k=>+k.slice(1)).sort((a,b)=>a-b);
  line(s,ks.map(k=>[x(k),y(hn['d'+k])]),c);
  ks.forEach(k=>{dot(s,x(k),y(hn['d'+k]),c,3);
   if(i===ns.length-1)label(s,x(k)+5,y(hn['d'+k])+4,hn['d'+k].toFixed(1),c)})});
}
function drawGate(d){
 const s=document.getElementById('ch_gate');s.setAttribute('viewBox','0 0 430 180');s.innerHTML='';
 const a=d.gate.A_bar_series||[];
 if(a.length<2){label(s,20,30,'积累中…','#6b7684');return}
 const[ymin,ymax]=mkY(a,0.12);const X=44,Y=14,W=370,H=140,n=a.length;
 frame(s,X,Y,W,H,n-1,ymin,ymax,'近期监督步','A_bar',[0,Math.floor(n/2),n-1]);
 line(s,a.map((v,i)=>[X+W*i/(n-1),Y+H-(v-ymin)/(ymax-ymin)*H]),'#6ee76e');
}
function drawQ(d){
 const s=document.getElementById('ch_q');s.setAttribute('viewBox','0 0 430 180');s.innerHTML='';
 const a=d.gate.q_hat_series||[];
 if(a.length<2){label(s,20,30,'续跑后记录（新监控项）…','#6b7684');return}
 const[ymin,ymax]=mkY(a,0.12);const X=44,Y=14,W=370,H=140,n=a.length;
 frame(s,X,Y,W,H,n-1,ymin,ymax,'近期监督步','q_hat',[0,Math.floor(n/2),n-1]);
 line(s,a.map((v,i)=>[X+W*i/(n-1),Y+H-(v-ymin)/(ymax-ymin)*H]),'#d2a8ff');
}
function drawTP(d){
 const s=document.getElementById('ch_tp');s.setAttribute('viewBox','0 0 430 180');s.innerHTML='';
 const f=d.throughput.fine||[];
 if(f.length>=2){
  const vals=f.map(p=>p.tok_s);
  const[ymin,ymax]=mkY(vals,0.12);
  const X=44,Y=14,W=370,H=140,n=f.length;
  frame(s,X,Y,W,H,n-1,ymin,ymax,'近期 batch','监督 tok/s',[0,Math.floor(n/2),n-1]);
  line(s,f.map((p,i)=>[X+W*i/(n-1),Y+H-(p.tok_s-ymin)/(ymax-ymin)*H]),'#79b8ff');
  label(s,X+W-78,Y+13,'实时 '+vals[vals.length-1]+' tok/s','#79b8ff');
 }else{
  const c=(d.throughput.coarse||[]);
  if(c.length<2){label(s,20,30,'细粒度数据积累中（逐 batch 时间戳）…','#6b7684');return}
  const vals=c.map(p=>p.tok_s);const[ymin,ymax]=mkY(vals,0.15);
  const X=44,Y=14,W=370,H=140,n=c.length;
  frame(s,X,Y,W,H,n-1,ymin,ymax,'验收周期','监督 tok/s',[0,Math.floor(n/2),n-1]);
  line(s,c.map((p,i)=>[X+W*i/(n-1),Y+H-(p.tok_s-ymin)/(ymax-ymin)*H]),'#79b8ff');
  label(s,X+W-70,Y+13,c[c.length-1].tok_s+' tok/s','#79b8ff');
 }
}
function drawB(d){
 const s=document.getElementById('ch_b');s.setAttribute('viewBox','0 0 860 170');s.innerHTML='';
 const pb=d.gate.per_b_lm_recent||{};const ks=Object.keys(pb);
 if(!ks.length){label(s,20,30,'积累中…','#6b7684');return}
 const mx=Math.max(...Object.values(pb));const X=52,Y=12,W=780,H=130;
 frame(s,X,Y,W,H,Math.max(...ks.map(Number)),0,mx*1.06,'监督深度 b','lm',ks.map(Number));
 ks.forEach(k=>{const h=H*(pb[k]/(mx*1.06)),xx=X+W*k/Math.max(...ks.map(Number));
  s.appendChild(el('rect',{x:xx-11,y:Y+H-h,width:22,height:h,fill:'#3d7edb',opacity:.85}));
  if(k%4===0)label(s,xx-8,Y+H-h-4,pb[k].toFixed(2),'#9aa4af')});
}
function card(k,v,d2){return `<div class="card stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d2||''}</div></div>`}
async function refresh(){
 let d;
 try{d=await (await fetch('/api/progress')).json()}catch(e){
  document.getElementById('chip').textContent='API 不可达';return}
 const p=d.progress,st=d.status;
 const chip=document.getElementById('chip');
 chip.textContent=st;chip.className='chip '+st;
 document.getElementById('subinfo').textContent=
  `Phase ${p.phase} · 服务时间 ${d.server_time} · 只读看板，与训练进程隔离`;
 const note=document.getElementById('paused_note');
 if(st!=='RUNNING'){
  const age=d.history_file?Math.round(d.history_file.age_sec/60):null;
  note.style.display='block';
  note.innerHTML=(st==='PAUSED'
   ? `⏸ <b>训练未在运行</b> — 本页全部指标为 <b>step ${p.step}</b> 暂停时的存档快照${age!=null?`（history ${age} 分钟前最后更新）`:''}。速度/ETA/A_bar/梯度等均为暂停瞬间的旧值，<b>不随等待时间变化</b>；续跑后自动恢复实时刷新。`
   : `✔ 训练已完成 — 以下为最终存档数据。`);
 }else{note.style.display='none'}
 const g=d.gate,lc=d.depth_curves.evals.slice(-1)[0];
 const frozen=st!=='RUNNING'?'<span style="color:#ffd35e">（冻结）</span>':'';
 const gain=p.depth_gain?`${p.depth_gain.d1_minus_d4>=0?'+':''}${p.depth_gain.d1_minus_d4}（对基线 ${p.depth_gain.vs_baseline_gain>=0?'+':''}${p.depth_gain.vs_baseline_gain}）`:'-';
 document.getElementById('stats').innerHTML=
  card('step / 总步数',`${p.step} / 1952`, p.records!=null&&p.records!==p.step
       ? `日志 ${p.records} 行（含重启重跑 ${p.records-p.step}）` : `进度 ${(p.fraction*100).toFixed(2)}%`) +
  card('监督 token',`${(p.sup_tokens/1e6).toFixed(3)}M`,`${(p.fraction*100).toFixed(2)}%`) +
  card('速度',`${p.sec_per_batch??'-'} s/b`,`≈${d.throughput.sup_tok_s??'-'} tok/s`+(frozen?'':' (实测)')) +
  card('ETA',p.eta_hours!=null?p.eta_hours+' h':'-',st==='RUNNING'?'至 5M':'暂停中不推进'+frozen) +
  card('深度增益 d1−d4',gain,'正=越循环越好') +
  card('门开度 A_bar',g.A_bar_mean_recent??'-','暂停时均值'+frozen) +
  card('置信头 q_hat',g.q_hat_mean??'-','续跑后训练'+frozen) +
  card('grad ‖·‖ max',g.grad_norm_recent_max??'-','检查点 '+(d.checkpoint?d.checkpoint.age_sec+'s 前':'-')+frozen);
 let bh='';(d.badges||[]).forEach(b=>{
  bh+=`<div class="badge"><span class="${b.ok?'ok':'bad'}">${b.ok?'✓':'✗'} ${b.name}</span><span class="k" style="color:#8b949e">${b.detail}</span></div>`});
 document.getElementById('badges').innerHTML=bh;
 let sh='';
 (d.samples&&d.samples.samples||[]).forEach(x=>{sh+=`<div class="samp">[d${x.depth} · ${x.n_tokens}tok] ${x.text.replace(/</g,'&lt;')}</div>`});
 document.getElementById('samples').innerHTML=sh||'<span class="k">续跑后每次评估生成 2 条</span>';
 drawHero(d);drawDepth(d);drawFrozen(d);drawNorm(d);drawGate(d);drawQ(d);drawTP(d);drawB(d);
 if(st!=='RUNNING'){
  label(document.getElementById('ch_gate'),250,168,`截至 step ${p.step} · 暂停快照`,'#ffd35e');
  label(document.getElementById('ch_q'),250,168,`截至 step ${p.step} · 暂停快照`,'#ffd35e');
  label(document.getElementById('ch_b'),620,160,`截至 step ${p.step} · 暂停快照`,'#ffd35e');
  label(document.getElementById('ch_tp'),250,168,`暂停前最后测量`,'#ffd35e');
 }
 document.getElementById('feed_train').innerHTML=(d.feeds.train||[]).map(t=>`<div>${t.replace(/</g,'&lt;')}</div>`).join('');
 document.getElementById('feed_acc').innerHTML=(d.feeds.accept||[]).map(t=>`<div class="${t.includes('verdict=PASS')?'hl':''}">${t.replace(/</g,'&lt;')}</div>`).join('');
 document.getElementById('foot').textContent=
  `history ${d.history_file?d.history_file.age_sec+'s 前更新':'-'} · 解析错误 ${d.parse_errors} · 数据源 runs/v5_l0/（只读）`;
}
async function loop(){try{await refresh()}catch(e){console.error(e)}}
loop();setInterval(loop,30000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    tail = None
    run_dir = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/progress"):
            body = json.dumps(self.tail.snapshot(self.run_dir),
                              ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        else:
            self.send_error(404)
            return
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8741)
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address; 0.0.0.0 exposes to the LAN (read-only metrics)")
    ap.add_argument("--run", default="v5_l0")
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run
    Handler.tail = Tail(run_dir / "history.jsonl")
    Handler.run_dir = run_dir
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard v2 (MiMo-style, read-only) -> http://{args.host}:{args.port}",
          flush=True)
    print(f"watching {run_dir / 'history.jsonl'}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
