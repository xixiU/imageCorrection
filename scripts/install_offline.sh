#!/usr/bin/env bash
# =============================================================================
# install_offline.sh —— 内网服务器离线安装脚本
#
# 本文件是源码，由 scripts/build_offline.sh 复制进离线包根目录（并去除 CRLF）。
# 不要在离线包里单独修改它，改这里再重新打包。
#
# 离线包目录结构（解压后）：
#   image-process-offline/
#   ├── install_offline.sh      本脚本
#   ├── BUILD_INFO.txt          构建信息（目标架构、Python 版本、wheel 清单）
#   ├── requirements.txt        依赖清单（与工程 requirements.txt 一致）
#   ├── offline_packages/       所有 wheel + SHA256SUMS
#   └── app/                    应用代码（app.py、image_processor.py、static/、README.md）
#
# 用法：
#   sudo bash install_offline.sh                              # 安装到 /opt/image-process
#   INSTALL_DIR=$HOME/image-process bash install_offline.sh   # 无 sudo 时装到家目录
#
# 可选环境变量：
#   INSTALL_DIR      安装目录，默认 /opt/image-process
#   PYTHON_BIN       指定 Python 3.8 解释器路径，默认依次尝试 python3.8、python3
#   SKIP_SMOKE_TEST  设为 1 跳过“启动服务并请求 /health”的冒烟测试
#   SMOKE_PORT       冒烟测试使用的临时端口，默认 18000（只监听 127.0.0.1）
#
# 执行步骤：
#   1. 环境检查：Linux、CPU 架构与离线包一致、glibc >= 2.17、Python 必须是 3.8
#   2. 校验 wheel 的 SHA256，防止传输损坏
#   3. 复制应用代码到 INSTALL_DIR（已有旧版本时先备份到 INSTALL_DIR/.backup/<时间>）
#   4. 创建虚拟环境 INSTALL_DIR/venv（系统缺 ensurepip 时自动用离线 pip 引导）
#   5. 只从 offline_packages/ 安装依赖（--no-index，不访问任何网络源）
#   6. 验证：pip check、逐个导入依赖、导入 app、启动服务请求 /health
#
# 建议保留安装日志：  sudo bash install_offline.sh 2>&1 | tee install.log
# =============================================================================
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="${PKG_DIR}/offline_packages"
APP_SRC="${PKG_DIR}/app"
BUILD_INFO="${PKG_DIR}/BUILD_INFO.txt"

INSTALL_DIR="${INSTALL_DIR:-/opt/image-process}"
PYTHON_BIN="${PYTHON_BIN:-}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"
SMOKE_PORT="${SMOKE_PORT:-18000}"
VENV_DIR="${INSTALL_DIR}/venv"

# wheel 是按 cp38 + manylinux(glibc 2.17) 下载的，这两项必须满足
REQUIRED_PY="3.8"
MIN_GLIBC="2.17"

log()  { printf '[INFO]  %s\n' "$*"; }
warn() { printf '[WARN]  %s\n' "$*" >&2; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
step() { printf '\n==== %s ====\n' "$*"; }

# -----------------------------------------------------------------------------
step "1/6 环境检查"
# -----------------------------------------------------------------------------
[ "$(uname -s)" = "Linux" ] || die "仅支持 Linux，当前系统: $(uname -s)"

for p in "$WHEEL_DIR" "$APP_SRC" "$BUILD_INFO" "${PKG_DIR}/requirements.txt"; do
    [ -e "$p" ] || die "离线包不完整，缺少: $p（请重新解压或重新传输）"
done

# CPU 架构必须与构建时的 TARGET_ARCH 一致，否则 numpy/opencv 等二进制 wheel 无法安装
build_arch="$(sed -n 's/^target_arch=//p' "$BUILD_INFO")"
host_arch="$(uname -m)"
[ "$build_arch" = "$host_arch" ] || \
    die "架构不匹配：离线包为 ${build_arch}，本机为 ${host_arch}。请在联网机器用 TARGET_ARCH=${host_arch} 重新打包"
log "CPU 架构: ${host_arch}"

# manylinux2014 wheel 要求 glibc >= 2.17（CentOS 7 恰好是 2.17）；Alpine 等 musl 系统无法使用
glibc_ver="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)"
[ -n "$glibc_ver" ] || die "未检测到 glibc（可能是 Alpine/musl 系统），manylinux wheel 无法使用"
if [ "$(printf '%s\n%s\n' "$MIN_GLIBC" "$glibc_ver" | sort -V | head -n1)" != "$MIN_GLIBC" ]; then
    die "glibc 版本 ${glibc_ver} 低于要求的 ${MIN_GLIBC}"
fi
log "glibc: ${glibc_ver}"

# 判断某个解释器是否为 Python 3.8
is_py38() {
    "$1" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 8) else 1)" >/dev/null 2>&1
}

PY=""
if [ -n "$PYTHON_BIN" ]; then
    # 用户显式指定时只认这一个，避免悄悄换成别的解释器
    is_py38 "$PYTHON_BIN" || die "PYTHON_BIN=${PYTHON_BIN} 不是 Python ${REQUIRED_PY}"
    PY="$PYTHON_BIN"
else
    for cand in python3.8 python3; do
        if command -v "$cand" >/dev/null 2>&1 && is_py38 "$cand"; then
            PY="$(command -v "$cand")"
            break
        fi
    done
fi
[ -n "$PY" ] || die "未找到 Python ${REQUIRED_PY}。离线包中的 wheel 均为 cp38，其他版本无法安装。
        可用 PYTHON_BIN=/path/to/python3.8 指定（例如 SCL: /opt/rh/rh-python38/root/usr/bin/python3.8）"
log "Python: ${PY} ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

# -----------------------------------------------------------------------------
step "2/6 校验 wheel 完整性"
# -----------------------------------------------------------------------------
[ -f "${WHEEL_DIR}/SHA256SUMS" ] || die "缺少 ${WHEEL_DIR}/SHA256SUMS"
(cd "$WHEEL_DIR" && sha256sum -c --quiet SHA256SUMS) || \
    die "wheel 校验失败，文件在传输中损坏，请重新传输离线包"
log "全部 $(wc -l < "${WHEEL_DIR}/SHA256SUMS") 个 wheel 校验通过"

# -----------------------------------------------------------------------------
step "3/6 部署应用代码到 ${INSTALL_DIR}"
# -----------------------------------------------------------------------------
mkdir -p "$INSTALL_DIR" 2>/dev/null || \
    die "无法创建 ${INSTALL_DIR}。请用 sudo 执行，或设置 INSTALL_DIR=\$HOME/image-process"
[ -w "$INSTALL_DIR" ] || \
    die "对 ${INSTALL_DIR} 无写权限。请用 sudo 执行，或设置 INSTALL_DIR=\$HOME/image-process"

# 已有旧版本：先备份除 venv 外的文件，再覆盖。回滚时把备份目录内容拷回即可
if [ -f "${INSTALL_DIR}/app.py" ]; then
    backup_dir="${INSTALL_DIR}/.backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    for item in "$INSTALL_DIR"/*; do
        [ "$(basename "$item")" = "venv" ] && continue
        cp -r "$item" "$backup_dir/"
    done
    log "检测到旧版本，已备份到 ${backup_dir}"
fi

cp -r "${APP_SRC}/." "$INSTALL_DIR/"
log "应用代码已复制"

# -----------------------------------------------------------------------------
step "4/6 准备虚拟环境 ${VENV_DIR}"
# -----------------------------------------------------------------------------
VPY="${VENV_DIR}/bin/python"
# 所有 pip 操作只看本地目录，不访问任何源（即使机器上配置了 pip.conf 镜像）
PIP_OFFLINE=(--no-index --find-links "$WHEEL_DIR" --disable-pip-version-check)

if [ -x "$VPY" ]; then
    is_py38 "$VPY" || die "已有虚拟环境 ${VENV_DIR} 不是 Python ${REQUIRED_PY}，请先手动移走它再重装"
    log "复用已有虚拟环境"
else
    # 目录存在但不是可用 venv（可能是用户自己的东西），不擅自删除
    [ ! -e "$VENV_DIR" ] || die "${VENV_DIR} 已存在但不是可用的虚拟环境，请先手动检查并移走"
    if "$PY" -m venv "$VENV_DIR"; then
        log "虚拟环境已创建"
    else
        # Debian/Ubuntu 未安装 python3.8-venv 时 ensurepip 不可用，内网又装不了这个包。
        # 改为不带 pip 创建 venv，再直接用离线包里的 pip wheel 给 venv 装 pip。
        warn "venv 创建失败（多为缺少 python3.8-venv / ensurepip），改用离线 pip 引导"
        rm -rf "$VENV_DIR"   # 只删除本次刚创建失败的目录（上面已确认之前不存在）
        "$PY" -m venv --without-pip "$VENV_DIR"
        pip_whl="$(ls "$WHEEL_DIR"/pip-*.whl 2>/dev/null | head -n1 || true)"
        [ -n "$pip_whl" ] || die "离线包中没有 pip wheel，无法引导"
        "$VPY" "${pip_whl}/pip" install "${PIP_OFFLINE[@]}" pip
        log "已通过离线 pip wheel 引导虚拟环境"
    fi
fi

# -----------------------------------------------------------------------------
step "5/6 离线安装依赖"
# -----------------------------------------------------------------------------
# 先升级 pip：Python 3.8 早期版本自带的 pip 19.2 不识别 manylinux2014 标签，
# 会报 "is not a supported wheel on this platform"
"$VPY" -m pip install "${PIP_OFFLINE[@]}" --upgrade pip
"$VPY" -m pip install "${PIP_OFFLINE[@]}" -r "${PKG_DIR}/requirements.txt"
log "依赖安装完成"

# -----------------------------------------------------------------------------
step "6/6 验证安装"
# -----------------------------------------------------------------------------
"$VPY" -m pip check || die "pip check 发现依赖冲突"

# 按 requirements.txt 逐个导入，依赖增减时无需改脚本。
# 包名与导入名不同的在 IMPORT_NAMES 里登记。
log "逐个导入依赖："
if ! "$VPY" - "${PKG_DIR}/requirements.txt" <<'PYCODE'
import importlib
import re
import sys

IMPORT_NAMES = {
    "opencv-python-headless": ["cv2"],
    "opencv-python": ["cv2"],
    "python-multipart": ["python_multipart", "multipart"],  # 0.0.13 起改名 python_multipart
    "pillow": ["PIL"],
}

failed = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.match(r"[A-Za-z0-9_.\-]+", line).group(0).lower()
        candidates = IMPORT_NAMES.get(name, [name.replace("-", "_")])
        for mod in candidates:
            try:
                m = importlib.import_module(mod)
            except ImportError:
                continue
            print("  [OK]   %-24s import %-18s %s" % (name, mod, getattr(m, "__version__", "")))
            break
        else:
            failed.append(name)
            print("  [FAIL] %-24s tried: %s" % (name, ", ".join(candidates)))

sys.exit(1 if failed else 0)
PYCODE
then
    die "存在无法导入的依赖，见上方 [FAIL]"
fi

# 导入 app 会连带导入 image_processor，能提前发现代码与依赖不匹配
(cd "$INSTALL_DIR" && "$VPY" -c "import app") || die "import app 失败，请检查应用代码"
log "import app 成功"

if [ "$SKIP_SMOKE_TEST" = "1" ]; then
    warn "已跳过冒烟测试（SKIP_SMOKE_TEST=1）"
else
    # 端口被占用时直接提示，避免把别的服务的响应误判为成功
    if ! "$VPY" -c "import socket,sys; s=socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1])))" "$SMOKE_PORT" 2>/dev/null; then
        die "冒烟测试端口 ${SMOKE_PORT} 已被占用。改用 SMOKE_PORT=其他端口，或 SKIP_SMOKE_TEST=1 跳过"
    fi

    smoke_log="${INSTALL_DIR}/.smoke_test.log"
    (cd "$INSTALL_DIR" && exec "$VPY" -m uvicorn app:app --host 127.0.0.1 --port "$SMOKE_PORT") \
        >"$smoke_log" 2>&1 &
    smoke_pid=$!
    trap 'kill "$smoke_pid" 2>/dev/null || true' EXIT

    # 内网机器常配置 http_proxy，这里显式禁用代理，保证请求直达本机
    health_ok=0
    for _ in $(seq 1 30); do
        if "$VPY" - "$SMOKE_PORT" >/dev/null 2>&1 <<'PYCODE'
import json, sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
url = "http://127.0.0.1:%s/health" % sys.argv[1]
body = json.load(opener.open(url, timeout=2))
sys.exit(0 if body.get("status") == "ok" else 1)
PYCODE
        then
            health_ok=1
            break
        fi
        kill -0 "$smoke_pid" 2>/dev/null || break   # 服务进程已退出，不必再等
        sleep 1
    done

    kill "$smoke_pid" 2>/dev/null || true
    wait "$smoke_pid" 2>/dev/null || true
    trap - EXIT

    if [ "$health_ok" = "1" ]; then
        log "冒烟测试通过：GET /health 返回 status=ok"
        rm -f "$smoke_log"
    else
        warn "服务日志（${smoke_log}）末尾："
        tail -n 30 "$smoke_log" >&2 || true
        die "冒烟测试失败：30 秒内 /health 未返回 status=ok"
    fi
fi

# -----------------------------------------------------------------------------
cat <<EOF

============================================================
 安装完成
   安装目录:   ${INSTALL_DIR}
   虚拟环境:   ${VENV_DIR}

 前台启动（必须在安装目录下执行）：
   cd ${INSTALL_DIR} && ${VPY} -m uvicorn app:app --host 0.0.0.0 --port 8000

 后台运行（systemd）、健康检查、curl 测试：
   见 ${INSTALL_DIR}/README.md
EOF
if [ "$(id -u)" = "0" ]; then
    cat <<EOF

 注意：本次以 root 安装，文件属主为 root。
   用专用账号运行服务前请执行：  chown -R <服务账号>:<服务账号> ${INSTALL_DIR}
EOF
fi
echo "============================================================"
