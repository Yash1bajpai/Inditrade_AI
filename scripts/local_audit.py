#!/usr/bin/env python3
"""
Local standing audit for IndiTrade AI.

Runs the OpenCode CLI against the local agentrouter proxy (127.0.0.1:8089 —
started via %USERPROFILE%\\.agentrouter\\start_proxy_background.vbs) to audit
the repository with GLM-5.3. Mirrors the CI workflow's logic but runs on the
developer's machine, where agentrouter.org is fast (GitHub-hosted runners are
WAF-tarpitted by the upstream).

One persistent OpenCode session ("IndiTrade AI Audit") is created on the first
run and reused forever after; its id lives in .audit/session_id.txt.
Reports are written to reports/ai_audit/<timestamp>.md with a rolling copy at
reports/ai_audit_latest.md. Read-only is enforced by the `plan` agent.

Schedule: Task Scheduler "IndiTrade AI Local Audit" (daily 21:00), or run
`python scripts/local_audit.py` any time.
"""

import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKER = REPO_ROOT / ".audit" / "session_id.txt"
REPORT_DIR = REPO_ROOT / "reports" / "ai_audit"
OC_DATA = Path.home() / ".local" / "share" / "opencode"
OC_DB = OC_DATA / "opencode.db"
SESSION_TITLE = "IndiTrade AI Audit"
MODEL = "agentrouter/glm-5.3"

BOOTSTRAP_PROMPT = (
    "You are the standing external auditor of the IndiTrade AI repository "
    "(Indian trade intelligence platform: FastAPI backend, Next.js frontend, "
    "ML models, hybrid BM25+vector RAG). This run BOOTSTRAPS your persistent "
    "session — keep it small and fast. Budget: at most 12 tool calls. Do "
    "exactly this: (1) run git log --oneline -20; (2) read README.md and "
    "PROJECT_MEMORY.md; (3) run git status and check no secrets/keys are "
    "tracked (git ls-files | grep -iE \"env|secret|key\" is enough); (4) skim "
    "src/backend/main.py and src/rag/sparse_index.py. Do NOT read anything "
    "else. Read-only: never modify files. Then write a markdown report under "
    "500 words: ## Summary, ## Findings (each tagged "
    "[CRITICAL]/[HIGH]/[MEDIUM]/[LOW] with file paths), ## Recommendations "
    "(max 5). Remember what you learn; later runs will only ask you to audit "
    "new commits."
)

INCREMENTAL_PROMPT = (
    "You are the same standing auditor of the IndiTrade AI repository, "
    "continuing your session. Keep it fast: under 10 minutes, at most 10 tool "
    "calls. Run git log --oneline -15, inspect only the recent changes (git "
    "show --stat and targeted file reads; read-only, never modify). Audit "
    "what changed plus possible regressions. Finish with a markdown report "
    "under 400 words: ## Summary, ## Findings (each tagged "
    "[CRITICAL]/[HIGH]/[MEDIUM]/[LOW] with file paths), ## Recommendations. "
    "If nothing needs attention, say so explicitly."
)


def load_session_id() -> str:
    if MARKER.exists():
        return MARKER.read_text(encoding="utf-8").strip()
    return ""


def session_exists(sid: str) -> bool:
    if not (OC_DB.exists() and sid):
        return False
    try:
        con = sqlite3.connect(f"file:{OC_DB}?mode=ro", uri=True)
        row = con.execute(
            "SELECT COUNT(*) FROM session WHERE id=?", (sid,)
        ).fetchone()
        con.close()
        return bool(row and row[0] == 1)
    except Exception:
        return False


def main() -> int:
    opencode = shutil.which("opencode")
    if not opencode:
        print("ERROR: opencode CLI not found on PATH.", file=sys.stderr)
        return 2

    sid = load_session_id()
    incremental = bool(sid and session_exists(sid))

    cmd = [opencode, "run", "--agent", "plan", "-m", MODEL, "--auto"]
    if incremental:
        cmd += ["-s", sid]
    else:
        cmd += ["--title", SESSION_TITLE, "--variant", "high"]
    cmd.append(INCREMENTAL_PROMPT if incremental else BOOTSTRAP_PROMPT)

    mode = "incremental (session %s)" % sid if incremental else "bootstrap"
    print(f"[*] Starting {mode} audit at {datetime.now():%Y-%m-%d %H:%M:%S}")

    started = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=5400)
    except subprocess.TimeoutExpired:
        print("ERROR: audit exceeded 90 minutes — treating as failed so the "
              "scheduled job never hangs on a stalled upstream.", file=sys.stderr)
        return 1
    elapsed = time.time() - started

    output = (result.stdout or "").strip()
    if result.returncode != 0 or not output:
        print(f"ERROR: audit failed (exit {result.returncode}).", file=sys.stderr)
        if result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)
        return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    report_path = REPORT_DIR / f"{stamp}.md"
    report_path.write_text(output + f"\n\n---\n*mode: {mode} | duration: {elapsed:.0f}s*\n", encoding="utf-8")
    (REPORT_DIR.parent / "ai_audit_latest.md").write_text(
        output + f"\n\n---\n*mode: {mode} | duration: {elapsed:.0f}s*\n", encoding="utf-8"
    )

    if not incremental:
        # New session was just created — persist its id for all future runs.
        # The DB may be locked by the opencode instance that just created the
        # session; retry briefly rather than silently losing the marker (a
        # lost marker would fork the persistent session on the next run).
        new_sid = ""
        for attempt in range(5):
            try:
                con = sqlite3.connect(f"file:{OC_DB}?mode=ro", uri=True)
                row = con.execute(
                    "SELECT id FROM session WHERE title=? ORDER BY time_updated DESC LIMIT 1",
                    (SESSION_TITLE,),
                ).fetchone()
                con.close()
                if row:
                    new_sid = row[0]
                break
            except sqlite3.OperationalError:
                time.sleep(2 * (attempt + 1))
        if new_sid:
            MARKER.parent.mkdir(parents=True, exist_ok=True)
            MARKER.write_text(new_sid, encoding="utf-8")
            print(f"[+] Session persisted: {new_sid}")
        else:
            print("WARNING: could not persist the new session id; the next "
                  "run will bootstrap a fresh session.", file=sys.stderr)

    print(f"[OK] Audit complete in {elapsed:.0f}s -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
