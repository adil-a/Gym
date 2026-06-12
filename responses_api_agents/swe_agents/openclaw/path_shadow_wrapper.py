#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys


ALWAYS_DENY = {
    "killall": "killall blocked: would terminate processes by name",
    "pkill": "pkill blocked: would terminate processes by pattern",
    "shutdown": "shutdown blocked: would terminate the runtime",
    "reboot": "reboot blocked: would terminate the runtime",
    "poweroff": "poweroff blocked: would terminate the runtime",
    "halt": "halt blocked: would terminate the runtime",
    "curl": "curl blocked: outbound HTTP from rollouts is denied (see wrapper.py)",
    "wget": "wget blocked: outbound HTTP from rollouts is denied (see wrapper.py)",
}


PATTERN_DENY = {
    "git": [
        (
            r"\bgit\b[^\n|;&]*?(?<![\"'=\w])(?:fetch|pull|clone|ls-remote)\b",
            "git network commands (fetch/pull/clone/ls-remote) blocked",
        ),
        (
            r"\bgit\s+remote\s+(?:add|set-url|set-head|update|rename|set-branches)\b",
            "git remote add/set-url/etc. blocked",
        ),
        (r"\bgit\s+submodule\s+(?:add|update|sync|init)\b", "git submodule add/update/sync/init blocked"),
        (r"\bgit\s+archive\b[^\n|;&]*\s--remote\b", "git archive --remote blocked"),
        (r"\bgit\b[^\n|;&]*?(?<![\"'=\w])(?:https?|git|ssh|ftp|ftps)://", "git command with remote URL blocked"),
        (r"\bgit\b[^\n|;&]*?(?<![\"'=\w])git@[\w.\-]+:", "git command with git@host: URL blocked"),
        (
            r"\bgit\s+\S+\b[^\n|;&]*\b(?:origin|upstream|remotes/[^\s/]+)/[\w./\-]+",
            "git command referencing remote-tracking ref blocked",
        ),
    ],
    "rm": [
        (r"\brm\s+(-\w+\s+)*(/\s*$|/\*)", "rm -rf / or rm -rf /* blocked"),
        (
            r"\brm\s+(-\w+\s+)*(/(bin|usr|etc|var|home|root|opt|lib|lib64|sbin|boot|dev|proc|sys))\s*$",
            "rm of critical system directories blocked",
        ),
    ],
    "kill": [
        (r"\bkill\s+.*\$\(", "kill with command substitution $(...) blocked"),
        (r"\bkill\s+.*`", "kill with backtick command substitution blocked"),
        (r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*\$\w+", "kill with shell variables blocked"),
        (r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*-1\b", "kill -1 (kill all user processes) blocked"),
        (r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*0\b", "kill 0 (kill process group) blocked"),
        (
            r"\bkill\s+(-[1-9]|-1[0-5]|-[A-Z]+|-SIG[A-Z]+)?\s*-([2-9]\d\d+|[1-9]\d\d\d+)\s*$",
            "kill with negative PID (process group) blocked",
        ),
    ],
    "dd": [
        (
            r"\bdd\s+.*of=\s*(/dev/sd[a-z]|/dev/nvme\w*|/dev/hd[a-z]|/dev/null)\b",
            "dd to /dev/sd*/nvme*/hd*/null blocked",
        ),
    ],
    "tmux": [
        (r"\btmux\s+(kill-server|kill-session)\b", "tmux kill-server / kill-session blocked"),
    ],
    "init": [
        (r"\binit\s+[06]\b", "init 0 / init 6 blocked: would terminate the runtime"),
    ],
}


WRAPPED_COMMANDS = sorted(set(ALWAYS_DENY) | set(PATTERN_DENY))


def install(target_dir: str) -> int:
    """Create one symlink per wrapped basename inside ``target_dir``.

    Idempotent: replaces any pre-existing entry at the same name. Used by
    ``openclaw.sh`` at install time so the symlinks never need to live in
    the repo. ``target_dir`` is the install destination (typically
    ``$SETUP_DIR/bin``) which already contains ``wrapper.py``.
    """
    target = os.path.abspath(target_dir)
    if not os.path.isdir(target):
        print(f"ERROR: --install target not a directory: {target}", file=sys.stderr)
        return 2
    wrapper_path = os.path.join(target, "wrapper.py")
    if not os.path.isfile(wrapper_path):
        print(f"ERROR: wrapper.py must already exist at {wrapper_path} before --install", file=sys.stderr)
        return 2
    for name in WRAPPED_COMMANDS:
        link = os.path.join(target, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink("wrapper.py", link)  # relative target — survives bind-mounts
    print(f"Installed {len(WRAPPED_COMMANDS)} wrapper symlinks at {target}: {', '.join(WRAPPED_COMMANDS)}")
    return 0


def deny(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def exec_real(basename: str, rest):
    """exec the first PATH-resolved executable whose dir is not our own."""
    our_dir = os.path.dirname(os.path.realpath(__file__))
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        try:
            if os.path.realpath(d) == our_dir:
                continue
        except OSError:
            continue
        candidate = os.path.join(d, basename)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            os.execv(candidate, [basename] + rest)
    print(f"ERROR: real {basename} not found on PATH (excluding {our_dir})", file=sys.stderr)
    return 127


def resolve_invocation():
    """Determine ``(basename, rest)`` from ``sys.argv`` regardless of whether
    the wrapper was invoked via its symlink or explicitly as ``wrapper.py``.

    Symlink mode:   argv = ["/openclaw_setup/bin/git", "fetch", "origin"]
    Explicit mode:  argv = ["wrapper.py", "git", "fetch", "origin"]
    """
    invoked = os.path.basename(sys.argv[0]) if sys.argv else ""
    if invoked in {"wrapper.py", "wrapper"}:
        if len(sys.argv) < 2:
            return None, None
        return sys.argv[1], sys.argv[2:]
    return invoked, sys.argv[1:]


def main() -> int:
    # Install-time helpers (used by openclaw.sh). These are only reachable
    # via explicit ``python wrapper.py ...`` invocation — no symlink is
    # called ``--install`` or ``--list-names``.
    if len(sys.argv) >= 2 and sys.argv[1] == "--list-names":
        for name in WRAPPED_COMMANDS:
            print(name)
        return 0
    if len(sys.argv) >= 2 and sys.argv[1] == "--install":
        if len(sys.argv) < 3:
            print("ERROR: --install requires a target directory argument", file=sys.stderr)
            return 2
        return install(sys.argv[2])

    basename, rest = resolve_invocation()
    if basename is None:
        print(
            "ERROR: wrapper requires either a basename-named symlink invocation or a basename as argv[1]",
            file=sys.stderr,
        )
        return 2

    if basename in ALWAYS_DENY:
        return deny(ALWAYS_DENY[basename])

    patterns = PATTERN_DENY.get(basename)
    if patterns is None:
        print(f"ERROR: wrapper: no rules registered for {basename!r}", file=sys.stderr)
        return 2

    cmd = basename + (" " + " ".join(rest) if rest else "")
    for pattern, msg in patterns:
        if re.search(pattern, cmd, flags=re.IGNORECASE):
            return deny(msg)

    return exec_real(basename, rest)


if __name__ == "__main__":
    sys.exit(main())
