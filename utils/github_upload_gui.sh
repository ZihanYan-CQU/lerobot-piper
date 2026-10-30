#!/usr/bin/env bash
# Reuse an available GUI environment, or prepare an isolated one without training dependencies.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
env_dir="$project_dir/.cache/github-upload-venv"
env_python="$env_dir/bin/python"

# The existing LeRobot environment may already provide the same GUI dependencies as the yzh tools.
candidates=(
    "${GITHUB_UPLOAD_PYTHON:-}"
    "$env_python"
    "$project_dir/.venv/bin/python"
    "$HOME/anaconda3/envs/lerobot/bin/python"
    "$HOME/miniconda3/envs/lerobot/bin/python"
    /usr/bin/python3
    "$(command -v python3)"
)
base_python=""
for candidate in "${candidates[@]}"; do
    [[ -x "$candidate" ]] || continue
    if "$candidate" -c 'import tkinter, customtkinter; from PIL import Image' 2>/dev/null; then
        exec "$candidate" "$script_dir/github_upload_gui.py" "$@"
    fi
    if [[ -z "$base_python" ]] && "$candidate" -c 'import tkinter' 2>/dev/null; then
        base_python="$candidate"
    fi
done
if [[ -z "$base_python" ]]; then
    echo '未找到带 Tkinter 的 Python。Ubuntu / Debian 请安装 python3-tk，或设置 GITHUB_UPLOAD_PYTHON。' >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="$project_dir/.cache/uv-github-upload"
    exec uv run --no-project --python "$base_python" \
        --with 'customtkinter>=5.2.2,<6' --with 'Pillow>=10,<13' \
        python "$script_dir/github_upload_gui.py" "$@"
fi

echo '首次启动：正在准备独立的 CustomTkinter 界面环境……'
# Use a fresh directory if a cache from another Python version is already present.
env_dir="$project_dir/.cache/github-upload-runtime"
env_python="$env_dir/bin/python"
if [[ -x "$env_python" ]] && "$env_python" -c 'import tkinter, customtkinter; from PIL import Image' 2>/dev/null; then
    exec "$env_python" "$script_dir/github_upload_gui.py" "$@"
fi
"$base_python" -m venv --without-pip "$env_dir"
if ! python3 -m pip --version >/dev/null 2>&1; then
    echo '需要 uv 或支持 --python 的 pip 来安装界面依赖，请安装其中一个后重试。' >&2
    exit 1
fi
python3 -m pip --python "$env_python" --cache-dir "$project_dir/.cache/pip-github-upload" \
    install 'customtkinter>=5.2.2,<6' 'Pillow>=10,<13'
exec "$env_python" "$script_dir/github_upload_gui.py" "$@"
