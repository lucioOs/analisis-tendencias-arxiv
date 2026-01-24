# src/merge_live_into_history.py
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


LIVE_PARQUET = Path("data/processed/live_dataset.parquet")
HISTORY_DIR = Path("data/history")
HISTORY_PARQUET = HISTORY_DIR / "history_union.parquet"

HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def log(level: str, msg: str) -> None:
    # ASCII only (Windows-safe)
    print(f"[{level}] {msg}", flush=True)


def make_uid(row: pd.Series) -> str:
    # Prefer stable IDs
    u = str(row.get("uid") or "").strip()
    if u:
        return u

    url = str(row.get("url") or "").strip()
    if url:
        return "url::" + url

    # Fallback: hash(date+text+source)
    base = f"{row.get('date')}|{row.get('source')}|{row.get('text')}"
    h = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()
    return "sha1::" + h


def normalize(df: pd.DataFrame, default_source: str) -> pd.DataFrame:
    df = df.copy()

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("published")
    text_col = cols.get("text") or cols.get("content") or cols.get("body")
    source_col = cols.get("source")
    url_col = cols.get("url")
    uid_col = cols.get("uid")

    if not date_col or not text_col:
        raise ValueError("Input must contain date/text (or published/content equivalents).")

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})
    if source_col and source_col != "source":
        df = df.rename(columns={source_col: "source"})
    if url_col and url_col != "url":
        df = df.rename(columns={url_col: "url"})
    if uid_col and uid_col != "uid":
        df = df.rename(columns={uid_col: "uid"})

    if "source" not in df.columns:
        df["source"] = default_source
    if "url" not in df.columns:
        df["url"] = None
    if "uid" not in df.columns:
        df["uid"] = None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["text"] = df["text"].astype(str)
    df["source"] = df["source"].astype(str)

    df = df.dropna(subset=["date", "text"]).copy()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    df["uid"] = df.apply(make_uid, axis=1)
    df = df.drop_duplicates(subset=["uid"]).reset_index(drop=True)
    return df[["uid", "date", "text", "source", "url"]]


def main():
    ap = argparse.ArgumentParser(description="Merge Live into Historical union (dedupe + cutoff)")
    ap.add_argument("--cutoff_days", type=int, default=7, help="Only move items older than N days into history")
    ap.add_argument("--max_rows_live", type=int, default=200000, help="Safety cap for live parquet")
    args = ap.parse_args()

    if not LIVE_PARQUET.exists():
        log("ERROR", f"Missing {LIVE_PARQUET}. Run live ingest/export first.")
        raise SystemExit(1)

    live = pd.read_parquet(LIVE_PARQUET)
    if live.empty:
        log("WARN", "Live dataset is empty. Nothing to merge.")
        return

    if len(live) > args.max_rows_live:
        log("WARN", f"Live has {len(live)} rows, truncating to last {args.max_rows_live} for safety.")
        live = live.sort_values("date").tail(args.max_rows_live)

    live = normalize(live, default_source="live")

    # Cutoff (mature items only)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=args.cutoff_days)
    to_move = live[live["date"] <= cutoff].copy()
    if to_move.empty:
        log("INFO", "No mature items to merge yet (cutoff not reached).")
        return

    # Load existing history (if any)
    if HISTORY_PARQUET.exists():
        hist = pd.read_parquet(HISTORY_PARQUET)
        hist = normalize(hist, default_source="history")  # normalize just in case
    else:
        hist = pd.DataFrame(columns=["uid", "date", "text", "source", "url"])

    before = len(hist)
    merged = pd.concat([hist, to_move], ignore_index=True)
    merged = merged.drop_duplicates(subset=["uid"]).sort_values("date").reset_index(drop=True)
    after = len(merged)

    # Save
    merged.to_parquet(HISTORY_PARQUET, index=False)

    log("INFO", f"History updated: {HISTORY_PARQUET}")
    log("INFO", f"Added rows: {after - before} | Total history: {after}")
    log("INFO", f"Merged window: moved <= {cutoff.date()} (cutoff_days={args.cutoff_days})")


if __name__ == "__main__":
    main()
