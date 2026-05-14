#!/usr/bin/env bash
#
# prepare_env_bundle.sh
# 打包当前 CTNet 项目的运行环境与关键配置，方便迁移到另一台机器。
#
# 用法：
#   ./scripts/prepare_env_bundle.sh                  # 默认打包 ./bd 环境
#   ./scripts/prepare_env_bundle.sh /path/to/env     # 指定 conda 环境前缀
#
# 产生内容：
#   dist/
#     env_export.yml          - `conda env export --prefix` 输出
#     env_packed.tar.gz       - conda-pack 生成的二进制环境（若安装了 conda-pack）
#     project_bundle.tar.gz   - 代码与必须的数据目录打包（可选，根据需要添加）
#
# 迁移步骤（目标机）：
#   1. 解压 env_packed.tar.gz（或按 env_export.yml 重新创建环境）
#   2. 解压 project_bundle.tar.gz 到目标路径
#   3. 激活环境并验证 `python -m torch.utils.collect_env`

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_PREFIX="${PROJECT_ROOT}/bd"
ENV_PREFIX="${1:-$DEFAULT_ENV_PREFIX}"
DIST_DIR="${PROJECT_ROOT}/dist"

mkdir -p "${DIST_DIR}"

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo "[ERROR] 环境目录不存在：${ENV_PREFIX}" >&2
  exit 1
fi

echo "==> 导出 Conda 环境到 env_export.yml"
conda env export --prefix "${ENV_PREFIX}" > "${DIST_DIR}/env_export.yml"

if command -v conda-pack >/dev/null 2>&1; then
  echo "==> 使用 conda-pack 打包环境 (env_packed.tar.gz)"
  conda pack -p "${ENV_PREFIX}" -o "${DIST_DIR}/env_packed.tar.gz"
else
  echo "[INFO] 未检测到 conda-pack，跳过二进制打包。可执行：conda install conda-pack"
fi

echo "==> 可选：打包项目必要文件"
ARCHIVE_LIST=(
  "main_subject_specific.py"
  "CTNet_model.py"
  "ctnet.py"
  "utils.py"
  "scripts/"
  "configs/"
  "mymat_raw/"
  "BCICIV_2a_gdf/"
  "BCICIV_2b_gdf/"
  "true_labels/"
  "logs/phase3_training_plan.md"
)

PROJECT_ARCHIVE="${DIST_DIR}/project_bundle.tar.gz"
tar czf "${PROJECT_ARCHIVE}" -C "${PROJECT_ROOT}" "${ARCHIVE_LIST[@]}" 2>/dev/null || {
  echo "[WARN] 部分目录不存在，忽略上述警告即可。"
}

cat <<EOF
----------------------------------------
已生成迁移文件：
  ${DIST_DIR}/env_export.yml
  ${DIST_DIR}/env_packed.tar.gz   (若可用)
  ${PROJECT_ARCHIVE}

复制到笔记本后的建议步骤：
  1. 解压 project_bundle.tar.gz，替换/新建目标目录。
  2. 若存在 env_packed.tar.gz：
       mkdir -p ~/CTNet/bd
       tar -xf env_packed.tar.gz -C ~/CTNet/bd
       source ~/CTNet/bd/bin/activate
     若没有 env_packed.tar.gz：
       conda env create -p ~/CTNet/bd -f env_export.yml
  3. 在新环境中运行：
       python - <<'PY'
       import torch
       print(torch.__version__, torch.cuda.is_available())
       if torch.cuda.is_available():
           print(torch.cuda.get_device_name(0))
       PY
  4. 根据需要调整 CTNET_NUM_WORKERS 等环境变量后运行训练指令。
----------------------------------------
EOF
