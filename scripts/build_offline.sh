#!/usr/bin/env bash
# =============================================================================
# build_offline.sh —— 在联网机器上制作内网离线安装包
#
# 产物（位于工程目录 dist/ 下）：
#   dist/image-process-offline.tar.gz          离线包，拷到内网服务器
#   dist/image-process-offline.tar.gz.sha256   离线包校验值，传输后核对
#   dist/image-process-offline/                打包前的暂存目录（可删除）
#     ├── install_offline.sh    安装脚本（复制自 scripts/install_offline.sh，已去除 CRLF）
#     ├── BUILD_INFO.txt        构建信息
#     ├── requirements.txt
#     ├── offline_packages/     所有依赖 wheel + SHA256SUMS
#     └── app/                  应用代码
#
# 用法（在工程根目录或任意目录均可）：
#   bash scripts/build_offline.sh                        # 目标为 x86_64 服务器
#   TARGET_ARCH=aarch64 bash scripts/build_offline.sh    # 目标为 ARM 服务器（鲲鹏/飞腾等）
#
# 可选环境变量：
#   TARGET_ARCH   目标服务器 CPU 架构，x86_64（默认）或 aarch64，与内网机器 `uname -m` 一致
#   PYTHON        构建机上用来执行 pip 的 Python，默认自动查找 python3 / python
#
# 关键设计：
#   - 用 pip download 的 --platform/--python-version/--abi 参数“交叉下载”，
#     所以构建机可以是 Windows/macOS/任意 Python 版本，下载到的一定是
#     Linux + CPython 3.8 的 wheel。
#   - 平台标签只取 manylinux2014 及更早（glibc 2.17），保证 CentOS 7 也能装。
#   - --only-binary=:all:：只要 wheel，不要源码包。内网通常没有编译器，
#     源码包到了内网也装不上，宁可在这里就失败。
#   - 额外打包一个兼容 Python 3.8 的 pip wheel：内网 Python 3.8 自带的 pip
#     可能太旧（不识别 manylinux2014），或系统缺 ensurepip 无法创建带 pip 的 venv。
#   - 依赖清单只认工程根目录的 requirements.txt，本脚本不硬编码任何依赖。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PKG_NAME="image-process-offline"
DIST_DIR="${PROJECT_ROOT}/dist"
STAGE_DIR="${DIST_DIR}/${PKG_NAME}"
TARBALL="${DIST_DIR}/${PKG_NAME}.tar.gz"

TARGET_ARCH="${TARGET_ARCH:-x86_64}"
TARGET_PY_VERSION="3.8"
TARGET_PY_TAG="cp38"

log()  { printf '[INFO]  %s\n' "$*"; }
warn() { printf '[WARN]  %s\n' "$*" >&2; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
step() { printf '\n==== %s ====\n' "$*"; }

# 计算 sha256：Linux/Git Bash 用 sha256sum，macOS 用 shasum
sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$@"
    else
        shasum -a 256 "$@"
    fi
}

# 去掉 Windows 换行符。Windows 上编辑过的 .sh 带 CRLF 时，
# 到 Linux 上会报 "/usr/bin/env: bash\r: No such file or directory"
strip_cr() {
    local f
    for f in "$@"; do
        tr -d '\r' < "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
    done
}

# -----------------------------------------------------------------------------
step "1/6 检查构建环境"
# -----------------------------------------------------------------------------
case "$TARGET_ARCH" in
    x86_64|aarch64) ;;
    *) die "不支持的 TARGET_ARCH=${TARGET_ARCH}，可选 x86_64 / aarch64" ;;
esac

# 依次尝试可用的 Python。Windows 上 python3 可能只是应用商店的占位程序，
# 所以要真正执行一次才算找到
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -m pip --version >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
fi
[ -n "$PY" ] || die "未找到带 pip 的 Python，可用 PYTHON=/path/to/python 指定"
log "构建用 Python: $("$PY" --version 2>&1)，$("$PY" -m pip --version)"

# pip 20.3 之前不支持多个 --platform 参数
pip_major="$("$PY" -m pip --version | awk '{print $2}' | cut -d. -f1)"
if [ "$pip_major" -lt 21 ]; then
    die "构建机 pip 版本过旧（需要 >= 21），请先执行: $PY -m pip install --upgrade pip"
fi

# 必需文件。缺任何一个都说明工程还没开发完，不应该打包
REQUIRED_FILES=(
    "app.py"
    "image_processor.py"
    "requirements.txt"
    "static/index.html"
    "scripts/install_offline.sh"
)
for f in "${REQUIRED_FILES[@]}"; do
    [ -f "${PROJECT_ROOT}/${f}" ] || die "缺少必需文件: ${f}"
done
log "必需文件齐全"
log "目标平台: Linux ${TARGET_ARCH} / CPython ${TARGET_PY_VERSION} / glibc >= 2.17"

# -----------------------------------------------------------------------------
step "2/6 准备暂存目录 ${STAGE_DIR}"
# -----------------------------------------------------------------------------
# 只清理本脚本自己生成的暂存目录和产物
rm -rf "$STAGE_DIR" "$TARBALL" "${TARBALL}.sha256"
mkdir -p "${STAGE_DIR}/offline_packages" "${STAGE_DIR}/app"

# 应用代码：根目录所有 .py、static/、README.md（README 随包带上，内网可直接查阅部署说明）
cp "${PROJECT_ROOT}"/*.py "${STAGE_DIR}/app/"
cp -r "${PROJECT_ROOT}/static" "${STAGE_DIR}/app/"
[ -f "${PROJECT_ROOT}/README.md" ] && cp "${PROJECT_ROOT}/README.md" "${STAGE_DIR}/app/"
cp "${PROJECT_ROOT}/requirements.txt" "${STAGE_DIR}/app/"
find "${STAGE_DIR}/app" -type d -name "__pycache__" -prune -exec rm -rf {} +

cp "${PROJECT_ROOT}/requirements.txt" "${STAGE_DIR}/requirements.txt"
cp "${SCRIPT_DIR}/install_offline.sh" "${STAGE_DIR}/install_offline.sh"
strip_cr "${STAGE_DIR}/install_offline.sh" "${STAGE_DIR}/requirements.txt" "${STAGE_DIR}/app/requirements.txt"
chmod +x "${STAGE_DIR}/install_offline.sh" 2>/dev/null || true   # Windows 上可能无效，安装时用 bash 执行即可
log "应用代码与安装脚本已就位"

# -----------------------------------------------------------------------------
step "3/6 下载依赖 wheel（目标 ${TARGET_ARCH}）"
# -----------------------------------------------------------------------------
# 用相对路径传给 pip，避免 Windows Git Bash 下 /d/... 路径转换问题
cd "$STAGE_DIR"

PIP_TARGET_ARGS=(
    --only-binary=:all:
    --implementation cp
    --python-version "$TARGET_PY_VERSION"
    --abi "$TARGET_PY_TAG"
    --platform "manylinux2014_${TARGET_ARCH}"
    --platform "manylinux_2_17_${TARGET_ARCH}"
    --dest offline_packages
    --disable-pip-version-check
)
# aarch64 没有 manylinux1/2010，只有 x86_64 需要补充这两个更老的标签
if [ "$TARGET_ARCH" = "x86_64" ]; then
    PIP_TARGET_ARGS+=(--platform "manylinux2010_x86_64" --platform "manylinux1_x86_64")
fi

"$PY" -m pip download "${PIP_TARGET_ARGS[@]}" -r requirements.txt \
    || die "依赖下载失败。若提示 No matching distribution，说明该版本没有 ${TARGET_ARCH}/cp38 的 wheel，需调整 requirements.txt 的版本"

# pip 本身：--python-version 3.8 会让 pip 自动选出仍支持 3.8 的最新版本
"$PY" -m pip download "${PIP_TARGET_ARGS[@]}" pip || die "pip wheel 下载失败"

# -----------------------------------------------------------------------------
step "4/6 校验下载结果"
# -----------------------------------------------------------------------------
non_wheel="$(find offline_packages -type f ! -name '*.whl' || true)"
[ -z "$non_wheel" ] || die "存在非 wheel 文件（内网无法编译源码包）: ${non_wheel}"

wheel_count="$(find offline_packages -name '*.whl' | wc -l | tr -d ' ')"
[ "$wheel_count" -gt 0 ] || die "未下载到任何 wheel"
ls offline_packages/pip-*.whl >/dev/null 2>&1 || die "缺少 pip wheel"

(cd offline_packages && sha256 *.whl > SHA256SUMS)
log "共 ${wheel_count} 个 wheel，已生成 offline_packages/SHA256SUMS："
(cd offline_packages && ls -1 *.whl | sed 's/^/        /')

cat > BUILD_INFO.txt <<EOF
package=${PKG_NAME}
build_time=$(date '+%Y-%m-%d %H:%M:%S %z')
build_host=$(uname -s)-$(uname -m)
target_arch=${TARGET_ARCH}
target_python=${TARGET_PY_VERSION}
min_glibc=2.17
wheel_count=${wheel_count}
EOF

# -----------------------------------------------------------------------------
step "5/6 打包"
# -----------------------------------------------------------------------------
cd "$DIST_DIR"
tar -czf "${PKG_NAME}.tar.gz" "$PKG_NAME"
sha256 "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256"

# -----------------------------------------------------------------------------
step "6/6 校验离线包"
# -----------------------------------------------------------------------------
listing="$(tar -tzf "${PKG_NAME}.tar.gz")"
for f in install_offline.sh BUILD_INFO.txt requirements.txt \
         offline_packages/SHA256SUMS app/app.py app/image_processor.py app/static/index.html; do
    printf '%s\n' "$listing" | grep -qx "${PKG_NAME}/${f}" || die "离线包中缺少 ${f}"
done

packed_wheels="$(printf '%s\n' "$listing" | grep -c '\.whl$' || true)"
[ "$packed_wheels" = "$wheel_count" ] || die "离线包内 wheel 数量 ${packed_wheels} 与下载数量 ${wheel_count} 不一致"

bash -n "${STAGE_DIR}/install_offline.sh" || die "install_offline.sh 语法检查失败"
if grep -q $'\r' "${STAGE_DIR}/install_offline.sh"; then
    die "install_offline.sh 仍含 CRLF"
fi
log "离线包内容校验通过"

size="$(du -h "${PKG_NAME}.tar.gz" | cut -f1)"
cat <<EOF

============================================================
 离线包已生成
   文件:   ${TARBALL}  (${size})
   校验:   ${TARBALL}.sha256
   目标:   Linux ${TARGET_ARCH} + Python ${TARGET_PY_VERSION}，glibc >= 2.17

 内网服务器上：
   sha256sum -c ${PKG_NAME}.tar.gz.sha256
   tar -xzf ${PKG_NAME}.tar.gz
   cd ${PKG_NAME} && sudo bash install_offline.sh
============================================================
EOF
