# 权重分卷复原说明

本仓库权重以 95MB 分卷存储（规避 GitHub 100MB 单文件限制）。

复原原始分片（与 model.safetensors.index.json 配套使用）：

```bash
cd model-00001-of-00010.safetensors.split-* 所在目录
for prefix in $(ls *.safetensors.split-00 | sed 's/\.split-00$//'); do
  cat "${prefix}".split-* > "$prefix"
done
```

复原后目录即标准 HF 格式（model-*.safetensors + index + config + tokenizer）。
