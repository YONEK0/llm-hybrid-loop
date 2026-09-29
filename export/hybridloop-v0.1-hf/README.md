
## 权重分卷说明

GitHub 单文件限 100MB，故 10 个 safetensors 分片均以 95MB 卷存储（共 200 卷）。
复原标准 HF 格式：见 REASSEMBLE.md（一条 for 循环 cat 即可），复原后
`AutoModelForCausalLM.from_pretrained(DIR, trust_remote_code=True)` 直接可用。
