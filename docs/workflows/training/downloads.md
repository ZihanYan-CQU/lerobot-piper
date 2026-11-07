# 下载训练资源

[训练目录](README.md) · [环境准备](../setup/README.md)

在训练服务器的仓库根目录执行。按实际路径修改下面变量，然后只运行所需下载项；如 Hub 要求授权，先按[登录与排错](../../troubleshooting/README.md)配置访问。

```bash
DATASET_NAMESPACE='sukasu403'
DATASET_BASE_DIR='/home/yzh/具身智能/训练数据'
BASE_MODEL_ROOT='/home/yzh/具身智能/基座模型/Lerobot'
```

**D1：下载 PaliGemma 资源，排除 `.safetensors` 权重文件。**

```bash
uv run --no-sync hf download google/paligemma-3b-pt-224 \
  --local-dir "${BASE_MODEL_ROOT}/paligemma-3b-pt-224" \
  --exclude "*.safetensors"
```

下载目录与训练配方的 `--policy.text_tokenizer_name` 一致。`"*.safetensors"` 保留引号，交给下载工具匹配。

**D2：下载训练数据集的 `v1.0` 版本。**

```bash
uv run --no-sync hf download "${DATASET_NAMESPACE}/shape-matching_89_with_dagger_30" \
  --local-dir "${DATASET_BASE_DIR}/shape-matching_89_with_dagger_30" \
  --repo-type dataset \
  --revision v1.0
```

训练这份下载的数据时，将[训练配方](README.md)中的 `--dataset.repo_id` 改为 `"${DATASET_NAMESPACE}/shape-matching_89_with_dagger_30"`，替换原示例中的 `dagger_merged`；`--dataset.root` 已与上述下载目录一致。

**D3：下载 Pi0.5 基座模型。**

```bash
uv run --no-sync hf download lerobot/pi05_base \
  --local-dir "${BASE_MODEL_ROOT}/pi05_base"
```

基座模型保存在 `基座模型/Lerobot` 下；具体训练选择哪个起始模型，仍由执行命令中的 `--policy.pretrained_path` 指定。
