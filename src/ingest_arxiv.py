from __future__ import annotations

import argparse
import json
import time
from urllib.parse import urlencode

import feedparser
import requests

from live_store import connect, upsert_item


def log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def arxiv_url(query: str, max_results: int) -> str:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return "https://export.arxiv.org/api/query?" + urlencode(params)


def main():
    ap = argparse.ArgumentParser(description="Ingesta arXiv robusta (requests + feedparser)")
    ap.add_argument("--query", default="cat:cs.AI OR cat:cs.LG OR cat:cs.CL")
    ap.add_argument("--max_results", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    url = arxiv_url(args.query, args.max_results)
    log("INFO", f"arXiv URL: {url}")

    # 🔒 FIX DEFINITIVO: requests con verify=False
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "LiveTrendsBot/1.0"},
            timeout=30,
            verify=False,   # <<< CLAVE
        )
        r.raise_for_status()
        content = r.content
    except Exception as e:
        log("ERROR", f"No se pudo descargar arXiv: {e}")
        raise SystemExit(1)

    feed = feedparser.parse(content)
    if getattr(feed, "bozo", False):
        log("WARN", f"arXiv bozo: {getattr(feed, 'bozo_exception', None)}")

    con = connect()
    entries = getattr(feed, "entries", []) or []
    new_count = 0

    for e in entries:
        uid = e.get("id") or e.get("link")
        if not uid:
            continue

        title = (e.get("title", "") or "").replace("\n", " ").strip()
        summary = (e.get("summary", "") or "").replace("\n", " ").strip()
        published = e.get("published") or e.get("updated")
        link = e.get("link")

        item = {
            "uid": f"arxiv::{uid}",
            "source": "arxiv",
            "published": published,
            "title": title,
            "text": f"{title}. {summary}".strip(),
            "url": link,
            "raw_json": json.dumps(e, ensure_ascii=False),
        }

        if upsert_item(con, item):
            new_count += 1

    time.sleep(args.sleep)
    log("INFO", f"arXiv items={len(entries)} | nuevos={new_count}")


if __name__ == "__main__":
    main()
