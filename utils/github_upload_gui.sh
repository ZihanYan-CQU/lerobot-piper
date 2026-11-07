#!/usr/bin/env bash
# Launch the upload window directly through uv and the existing project environment.
set -euo pipefail
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
# Retired interpreter overrides must not bypass the uv project environment.
unset GITHUB_UPLOAD_PYTHON
export PYTHONNOUSERSITE=1

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
project_python="$project_dir/.venv/bin/python"

dependency_help() {
    printf '%s\n' \
        'GitHub 上传界面需要项目 desktop extra 中的 PyQt5。' \
        '请在项目根目录执行：uv sync --locked --inexact --extra desktop' \
        '同步时同时保留本机其他所需 extra；本启动器不会自动安装依赖。' >&2
}

if [[ ! -x "$project_python" ]]; then
    printf '项目环境不存在：%s\n' "$project_python" >&2
    dependency_help
    exit 1
fi
uv_bin="${PIPER_UV:-}"
if [[ -z "$uv_bin" ]]; then
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        uv_bin="$HOME/.local/bin/uv"
    else
        uv_bin="$(command -v uv || true)"
    fi
fi
if [[ ! -x "$uv_bin" ]]; then
    printf '%s\n' '未找到 uv，请先安装 uv。' >&2
    exit 1
fi

# Do not let an activated environment supply Python packages, native libraries or tools.
conda_roots=("$HOME/anaconda3" "$HOME/miniconda3")
if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_roots+=("$(dirname -- "$(dirname -- "$CONDA_EXE")")")
fi
for name in ${!CONDA_PREFIX@}; do
    [[ -z "${!name}" ]] || conda_roots+=("${!name}")
done
IFS=: read -r -a path_entries <<< "${PATH:-/usr/local/bin:/usr/bin:/bin}"
clean_path="$project_dir/.venv/bin"
for entry in "${path_entries[@]}"; do
    [[ -n "$entry" ]] || continue
    skip=false
    for prefix in "${conda_roots[@]}"; do
        if [[ "$entry" == "$prefix" || "$entry" == "$prefix/"* ]]; then
            skip=true
            break
        fi
    done
    [[ "$skip" == true ]] || clean_path+=":$entry"
done
export PATH="$clean_path"
unset VIRTUAL_ENV _CE_CONDA _CE_M
for name in ${!CONDA_@}; do unset "$name"; done
unset UV_PROJECT UV_WORKING_DIR UV_ENV_FILE UV_PYTHON UV_PYTHON_PREFERENCE
unset UV_NO_MANAGED_PYTHON UV_ISOLATED UV_NO_PROJECT UV_CONFIG_FILE UV_NO_CONFIG
export UV_PROJECT_ENVIRONMENT="$project_dir/.venv"

# Qt's bundled plugin path must stay absolute for this project's non-ASCII path.
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QML_IMPORT_PATH QML2_IMPORT_PATH
for qt_plugins in "$project_dir"/.venv/lib/python*/site-packages/PyQt5/Qt5/plugins; do
    if [[ -d "$qt_plugins/platforms" ]]; then
        export QT_PLUGIN_PATH="$qt_plugins"
        export QT_QPA_PLATFORM_PLUGIN_PATH="$qt_plugins/platforms"
        break
    fi
done

uv_command=("$uv_bin" run --project "$project_dir" --python "$project_python" --no-sync --no-env-file --)
if ! "${uv_command[@]}" python -c 'from PyQt5 import QtCore, QtGui, QtWidgets'; then
    dependency_help
    exit 1
fi
exec "${uv_command[@]}" python "$script_dir/github_upload_gui.py" "$@"
