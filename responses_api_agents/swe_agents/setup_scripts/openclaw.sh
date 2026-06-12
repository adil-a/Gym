#!/usr/bin/env bash
# One-time installer for the OpenClaw harness. Idempotent; safe to re-run.
# Cross-node-safe via a mkdir-based lock (same pattern as openhands.sh).
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"   # responses_api_agents/swe_agents/
SRC_DIR="$PARENT_DIR/openclaw"
SETUP_DIR="${OPENCLAW_SETUP_DIR_OVERRIDE:-$PARENT_DIR/swe_openclaw_setup}"

OPENCLAW_VERSION="${OPENCLAW_VERSION:-2026.5.6}"
NODE_VERSION="${NODE_VERSION:-22.22.3}"  # openclaw@2026.5.6 needs node >=22.14 (undici dep >=22.19)
PY_VERSION="${PY_VERSION:-3.12.7}"
PY_RELEASE="${PY_RELEASE:-20241016}"

mkdir -p "$SETUP_DIR"

export npm_config_cache="$SETUP_DIR/.npm-cache"
mkdir -p "$npm_config_cache"

# --- Cross-node mkdir-lock. 1h stale break. ---
LOCK="$SETUP_DIR/.openclaw_setup.lockdir"
WAITED=0
while ! mkdir "$LOCK" 2>/dev/null; do
  if [ -d "$LOCK" ]; then
    if [ "$(find "$LOCK" -maxdepth 0 -mmin +60 -print -quit 2>/dev/null)" = "$LOCK" ]; then
      echo "Breaking stale openclaw setup lock at $LOCK" >&2
      rm -rf "$LOCK"
      continue
    fi
  fi
  if [ "$WAITED" -ge 3600 ]; then
    echo "Timed out waiting for $LOCK" >&2
    exit 1
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  if [ $((WAITED % 30)) -eq 0 ]; then
    echo "Waiting for openclaw setup lock at $LOCK (${WAITED}s)..." >&2
  fi
done
trap 'rm -rf "$LOCK"' EXIT

# --- 1. Node (version-pinned; re-download if the on-disk node differs) ---
INSTALLED_NODE=""
if [ -x "$SETUP_DIR/node/bin/node" ]; then
  INSTALLED_NODE="$("$SETUP_DIR/node/bin/node" --version 2>/dev/null | sed 's/^v//')"
fi
if [ "$INSTALLED_NODE" != "$NODE_VERSION" ]; then
  echo "Downloading Node v${NODE_VERSION} (was: ${INSTALLED_NODE:-none})..."
  rm -rf "$SETUP_DIR/node"
  mkdir -p "$SETUP_DIR/node"
  curl -fsSL \
    "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
    | tar -xJ --strip-components=1 -C "$SETUP_DIR/node"
fi

# Put the bundled node on PATH for the rest of this script. node-based CLIs (npm, openclaw, npx)
# use `#!/usr/bin/env node`, so without this they fail with exit 127 on hosts lacking a system
# node. This does not change the version detection above — it just lets the (re)install run. The
# bundled node is also what rollouts use.
export PATH="$SETUP_DIR/node/bin:$PATH"

# --- 2. Standalone CPython ---
if [ ! -x "$SETUP_DIR/python/bin/python3" ]; then
  echo "Downloading CPython ${PY_VERSION}+${PY_RELEASE}..."
  curl -fsSL \
    "https://github.com/astral-sh/python-build-standalone/releases/download/${PY_RELEASE}/cpython-${PY_VERSION}+${PY_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz" \
    | tar -xz -C "$SETUP_DIR"
fi

# --- 3. Proxy venv ---
if [ ! -x "$SETUP_DIR/proxy_venv/bin/python" ]; then
  "$SETUP_DIR/python/bin/python3" -m venv "$SETUP_DIR/proxy_venv"
  "$SETUP_DIR/proxy_venv/bin/pip" install --no-cache-dir 'aiohttp>=3.9' 'jinja2>=3'
fi

# --- 4. OpenClaw via npm (project-local) ---
INSTALLED_VERSION=""
if [ -x "$SETUP_DIR/node_modules/.bin/openclaw" ]; then
  INSTALLED_VERSION="$("$SETUP_DIR/node_modules/.bin/openclaw" --version 2>/dev/null || true)"
fi
if [ "$INSTALLED_VERSION" != "$OPENCLAW_VERSION" ]; then
  pushd "$SETUP_DIR" >/dev/null
  if [ ! -f package.json ]; then
    ./node/bin/npm init -y >/dev/null
  fi
  ./node/bin/npm install --no-fund --no-audit "openclaw@${OPENCLAW_VERSION}"
  popd >/dev/null
fi

# --- 4b. Neutralize OpenClaw auto-compaction in the vendored bundle ---
# OpenClaw's context-summarization REPLACES conversation history mid-episode, which breaks the
# prompt_token_ids prefix-contiguity that on-policy RL training requires. The triggers are baked
# into dist/ (no config lever disables them in the pinned version), and npm regenerates dist/ on
# every reinstall, so we re-apply the surgical source patch here. Idempotent; fails loud if a
# marker is missing (i.e. OpenClaw changed shape) so the neutralization is never silently dropped.
"$SETUP_DIR/python/bin/python3" "$SRC_DIR/patch_openclaw.py" "$SETUP_DIR/node_modules/openclaw/dist"

# --- 5. Copy source files into setup dir ---
mkdir -p "$SETUP_DIR/bin"
cp -f "$SRC_DIR/stream_shim.py"             "$SETUP_DIR/stream_shim.py"
cp -f "$SRC_DIR/run_openclaw.sh"            "$SETUP_DIR/run_openclaw.sh"
cp -f "$SRC_DIR/path_shadow_wrapper.py"     "$SETUP_DIR/bin/wrapper.py"
# Pin the wrapper interpreter to OUR bundled python at its in-SIF mount path, so the command
# denylist never depends on the rollout SIF having a python3 on PATH. The source file keeps a
# portable `#!/usr/bin/env python3` (host/CI tests run it via that shebang); we stamp the in-SIF
# absolute path onto the installed copy that actually runs inside the SIF. /openclaw_setup is the
# fixed in-SIF mount of $SETUP_DIR (config_templates pathPrepend + app.py apptainer mounts).
sed -i '1s|.*|#!/openclaw_setup/python/bin/python3|' "$SETUP_DIR/bin/wrapper.py"
cp -f "$PARENT_DIR/prompts/openclaw/user_prompt.j2"   "$SETUP_DIR/user_prompt.j2"
# Language-parametrized gitignore script: exclude build artifacts from multilingual patches.
cp -f "$SRC_DIR/gitignore.sh" "$SETUP_DIR/gitignore.sh"
chmod +x "$SETUP_DIR/gitignore.sh"
chmod +x "$SETUP_DIR/run_openclaw.sh"

# --- 6. Install PATH-shadow symlinks ---
"$SETUP_DIR/python/bin/python3" "$SETUP_DIR/bin/wrapper.py" --install "$SETUP_DIR/bin"

# --- 7. Smoke checks (fail loud) ---
# Validate against the BUNDLED node (what rollouts use inside the testbed
# container), not whatever system node happens to be on PATH here.
PATH="$SETUP_DIR/node/bin:$PATH" "$SETUP_DIR/node_modules/.bin/openclaw" --version | grep -qF "$OPENCLAW_VERSION"
"$SETUP_DIR/proxy_venv/bin/python" -c 'import aiohttp, jinja2'
test -f "$SETUP_DIR/user_prompt.j2"
test -f "$SETUP_DIR/gitignore.sh"
# The compaction-neutralization patch must be present in the (re)generated bundle.
if ! grep -rqF "nemo-gym-no-compaction" "$SETUP_DIR/node_modules/openclaw/dist/"; then
  echo "ERROR: OpenClaw compaction-neutralization patch is not present in dist/" >&2
  exit 1
fi
# The unknown-tool-surfacing patch (fix #4) must land in BOTH name-acceptance predicates
# (record-time + model-facing replay) -> expect exactly two sentinel-bearing files in dist/.
SURFACE_TOOL_HITS="$(grep -rlF "nemo-gym-surface-unknown-tool" "$SETUP_DIR/node_modules/openclaw/dist/" | wc -l)"
if [ "$SURFACE_TOOL_HITS" -lt 2 ]; then
  echo "ERROR: OpenClaw unknown-tool-surfacing patch is incomplete (found $SURFACE_TOOL_HITS/2 sites)" >&2
  exit 1
fi
# The installed wrapper's shebang now points at the in-SIF python path (step 5), which does not
# exist on this host, so invoke it via the bundled interpreter explicitly to test the deny logic.
if "$SETUP_DIR/python/bin/python3" "$SETUP_DIR/bin/git" fetch 2>/dev/null; then
  echo "ERROR: PATH-shadow wrapper failed to deny 'git fetch'" >&2
  exit 1
fi
# Fail loud if the interpreter pin did not land (e.g. the source shebang shape changed).
if ! head -1 "$SETUP_DIR/bin/wrapper.py" | grep -qxF "#!/openclaw_setup/python/bin/python3"; then
  echo "ERROR: wrapper.py shebang was not pinned to the bundled in-SIF python" >&2
  exit 1
fi

echo "OpenClaw setup complete at $SETUP_DIR"
