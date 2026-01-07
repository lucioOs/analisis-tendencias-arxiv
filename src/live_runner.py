from __future__ import annotations

import subprocess
import sys


def log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out.strip()


def main() -> None:
    # 1) arXiv (fuente LIVE principal)
    code, out = run([
        sys.executable, "src/ingest_arxiv.py",
        "--query", "cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        "--max_results", "50"
    ])
    if code != 0:
        log("ERROR", f"arXiv falló.\n{out}")
        raise SystemExit(code)

    log("INFO", out)

    # 2) Export
    code, out = run([sys.executable, "src/export_live_dataset.py", "--days", "365"])
    if code == 2:
        log("WARN", out)
        return
    if code != 0:
        log("ERROR", f"Export falló.\n{out}")
        raise SystemExit(code)

    log("INFO", out)
    log("INFO", "✅ Live actualizado correctamente.")


if __name__ == "__main__":
    main()
