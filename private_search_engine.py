#!/usr/bin/env python3
"""
NexusSearch: A Privacy-First Hybrid & Metasearch Engine in Python
=================================================================
Designed for self-hosting or free deployment on Microsoft Azure App Service (F1 Free Tier)
with optional plug-and-play integration for Azure AI Search (Free Tier).

Key Resume / Architecture Highlights:
1. Hybrid Information Retrieval: Combines a local SQLite FTS5 inverted index (Okapi BM25)
   with asynchronous external API connectors (Wikipedia MediaWiki API, Hacker News Algolia API,
   GitHub REST Search API, and Microsoft Azure AI Search REST API).
2. Reciprocal Rank Fusion (RRF): Merges heterogeneous ranked lists using RRF (k=60), the same
   rank-aggregation algorithm used in Microsoft Azure AI Search hybrid retrieval.
3. Zero-Tracking Privacy Architecture: Strips client IPs, cookies, and referrers; enforces strict
   no-referrer security headers; and uses an in-memory TTL cache with zero query-IP logging.
4. Dual-Runtime Support: Runs on FastAPI + Uvicorn (with automatic OpenAPI docs at /docs) or falls
   back automatically to Python's built-in http.server for zero-dependency portability.

Quick Start:
    pip install fastapi uvicorn
    uvicorn private_search_engine:app --host 0.0.0.0 --port 8000

Azure App Service (Linux Free F1 Tier) Startup Command:
    gunicorn -w 2 -k uvicorn.workers.UvicornWorker private_search_engine:app --bind=0.0.0.0:8000
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import html
from html.parser import HTMLParser
import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Configuration & Environment Variables (Azure App Service Compatible)
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get("SEARCH_DB_PATH", "nexus_search_index.db")
CACHE_TTL_SECONDS = int(os.environ.get("SEARCH_CACHE_TTL", "300"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("SEARCH_API_TIMEOUT", "4.0"))
RRF_K = 60  # Standard Reciprocal Rank Fusion constant used in Azure AI Search

# Optional Microsoft Azure AI Search (Free Tier) Credentials
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_SEARCH_API_VERSION", "2024-07-01")

USER_AGENT = "NexusPrivateSearch/1.0 (+https://learn.microsoft.com/azure/app-service/)"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    source_rank: int
    raw_score: float = 0.0
    rrf_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lightweight HTML Text Extractor (for Local Crawler / Indexer)
# ---------------------------------------------------------------------------
class _HTMLTextExtractor(HTMLParser):
    """Extracts clean readable text and <title> from raw HTML without external dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self._ignore_tags = {"script", "style", "noscript", "svg", "head", "nav", "footer"}
        self._active_ignore = 0
        self._in_title = False
        self.title_parts: List[str] = []
        self.text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "title":
            self._in_title = True
        elif tag_lower in self._ignore_tags:
            self._active_ignore += 1

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "title":
            self._in_title = False
        elif tag_lower in self._ignore_tags and self._active_ignore > 0:
            self._active_ignore -= 1

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        elif self._active_ignore == 0:
            self.text_parts.append(cleaned)

    def get_extracted(self) -> Tuple[str, str]:
        title = " ".join(self.title_parts).strip()
        body = re.sub(r"\s+", " ", " ".join(self.text_parts)).strip()
        return title, body


# ---------------------------------------------------------------------------
# In-Memory Privacy Cache (TTL + Max Size, Zero User/IP Logging)
# ---------------------------------------------------------------------------
class PrivacyQueryCache:
    """Caches anonymous query results in memory for fast repeat lookups without logging users."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        entry = self._store.get(key)
        if not entry:
            return None
        timestamp, payload = entry
        if now - timestamp > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        return payload

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        if len(self._store) >= self.max_entries:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            self._store.pop(oldest_key, None)
        self._store[key] = (time.time(), payload)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Local Inverted Indexer & Okapi BM25 Engine (SQLite FTS5)
# ---------------------------------------------------------------------------
class LocalBM25Index:
    """
    Full-text search index backed by SQLite's FTS5 extension.
    Uses Porter stemming and Okapi BM25 scoring with custom column weights.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    url UNINDEXED,
                    title,
                    content,
                    tags,
                    indexed_at UNINDEXED,
                    tokenize = 'porter unicode61'
                );
                """
            )
            cursor = conn.execute("SELECT count(*) AS cnt FROM documents_fts")
            count = cursor.fetchone()["cnt"]
            if count == 0:
                self._seed_default_documents(conn)

    def _seed_default_documents(self, conn: sqlite3.Connection) -> None:
        """Seeds foundational technical documents so local BM25 search works immediately."""
        seed_docs = [
            (
                "https://learn.microsoft.com/en-us/azure/app-service/quickstart-python",
                "Deploy a Python (Django, Flask, or FastAPI) Web App to Azure App Service",
                "Learn how to deploy a Python FastAPI or Flask web application to Microsoft Azure App Service on Linux using the Free F1 pricing tier.",
                "microsoft azure python fastapi deployment cloud",
            ),
            (
                "https://learn.microsoft.com/en-us/azure/search/search-try-for-free",
                "Try Azure AI Search for Free - Microsoft Learn",
                "Create an Azure AI Search service on the Free tier to build hybrid search applications combining BM25 full-text ranking, vector search, and Reciprocal Rank Fusion (RRF).",
                "microsoft azure ai search bm25 rrf hybrid retrieval",
            ),
            (
                "https://sqlite.org/fts5.html",
                "SQLite FTS5 Full-Text Search Extension and Okapi BM25 Ranking",
                "FTS5 is an SQLite virtual table module that provides full-text search functionality for database applications, featuring built-in Okapi BM25 relevance scoring and snippet highlighting.",
                "sqlite fts5 bm25 inverted index python database",
            ),
            (
                "https://fastapi.tiangolo.com/",
                "FastAPI: High Performance Python Web Framework for APIs",
                "FastAPI is a modern, fast web framework for building REST APIs with Python based on standard Python type hints and OpenAPI.",
                "python fastapi rest api backend async openapi",
            ),
        ]
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for url, title, content, tags in seed_docs:
            conn.execute(
                "INSERT INTO documents_fts (url, title, content, tags, indexed_at) VALUES (?, ?, ?, ?, ?)",
                (url, title, content, tags, now_str),
            )
        conn.commit()

    def index_document(self, url: str, title: str, content: str, tags: str = "") -> Dict[str, Any]:
        """Inserts or updates a document in the local FTS5 inverted index."""
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._get_conn() as conn:
            conn.execute("DELETE FROM documents_fts WHERE url = ?", (url,))
            conn.execute(
                "INSERT INTO documents_fts (url, title, content, tags, indexed_at) VALUES (?, ?, ?, ?, ?)",
                (url.strip(), title.strip(), content.strip(), tags.strip(), now_str),
            )
            conn.commit()
        return {"status": "indexed", "url": url, "title": title, "indexed_at": now_str}

    def crawl_and_index_url(self, url: str, tags: str = "crawled") -> Dict[str, Any]:
        """Fetches a live web page anonymously, extracts clean text, and indexes it locally."""
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
        extractor = _HTMLTextExtractor()
        extractor.feed(raw_html)
        title, body = extractor.get_extracted()
        if not title:
            title = url
        return self.index_document(url=url, title=title[:200], content=body[:20000], tags=tags)

    @staticmethod
    def _sanitize_fts_query(raw_query: str) -> str:
        """Converts arbitrary user input into a safe FTS5 token query with prefix matching."""
        tokens = re.findall(r"[A-Za-z0-9_]+", raw_query.lower())
        if not tokens:
            return ""
        return " OR ".join(f'"{token}"*' for token in tokens[:10])

    def search(self, query: str, limit: int = 8) -> List[SearchResult]:
        """Queries the local FTS5 index and ranks matches using weighted Okapi BM25."""
        fts_query = self._sanitize_fts_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT
                url,
                title,
                snippet(documents_fts, 2, '<mark>', '</mark>', ' ... ', 28) AS snippet_html,
                tags,
                indexed_at,
                bm25(documents_fts, 0.0, 5.0, 1.0, 2.5) AS bm25_raw
            FROM documents_fts
            WHERE documents_fts MATCH ?
            ORDER BY bm25_raw ASC
            LIMIT ?
        """
        results: List[SearchResult] = []
        with self._get_conn() as conn:
            rows = conn.execute(sql, (fts_query, limit)).fetchall()
            for idx, row in enumerate(rows, start=1):
                positive_bm25 = round(abs(float(row["bm25_raw"])), 4)
                results.append(
                    SearchResult(
                        title=row["title"],
                        url=row["url"],
                        snippet=row["snippet_html"],
                        source="Local BM25 Index",
                        source_rank=idx,
                        raw_score=positive_bm25,
                        metadata={"tags": row["tags"], "indexed_at": row["indexed_at"]},
                    )
                )
        return results

    def stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            total = conn.execute("SELECT count(*) AS cnt FROM documents_fts").fetchone()["cnt"]
        return {"total_indexed_documents": total, "engine": "SQLite FTS5 (Okapi BM25)"}


# ---------------------------------------------------------------------------
# External Privacy-Preserving API Connectors
# ---------------------------------------------------------------------------
def _fetch_json(url: str, headers: Optional[Dict[str, str]] = None, post_body: Optional[Dict[str, Any]] = None) -> Any:
    """Executes an anonymous HTTP request with zero user cookies or client IP forwarding."""
    req_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)

    data_bytes = None
    method = "GET"
    if post_body is not None:
        data_bytes = json.dumps(post_body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
        method = "POST"

    req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def search_wikipedia_api(query: str, limit: int = 6) -> List[SearchResult]:
    """Queries the free MediaWiki Action API anonymously."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "utf8": "1",
            "format": "json",
            "srlimit": str(limit),
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    try:
        data = _fetch_json(url)
        items = data.get("query", {}).get("search", [])
        results: List[SearchResult] = []
        for idx, item in enumerate(items, start=1):
            title = item.get("title", "Untitled")
            page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            raw_snippet = item.get("snippet", "")
            clean_snippet = re.sub(r"<span class=\"searchmatch\">(.*?)</span>", r"<mark>\1</mark>", raw_snippet)
            results.append(
                SearchResult(
                    title=title,
                    url=page_url,
                    snippet=clean_snippet,
                    source="Wikipedia API",
                    source_rank=idx,
                    raw_score=round(1.0 / idx, 4),
                    metadata={"wordcount": item.get("wordcount", 0)},
                )
            )
        return results
    except Exception:
        return []


def search_hackernews_api(query: str, limit: int = 6) -> List[SearchResult]:
    """Queries the free Hacker News Algolia Search API anonymously."""
    params = urllib.parse.urlencode({"query": query, "tags": "story", "hitsPerPage": str(limit)})
    url = f"https://hn.algolia.com/api/v1/search?{params}"
    try:
        data = _fetch_json(url)
        hits = data.get("hits", [])
        results: List[SearchResult] = []
        for idx, hit in enumerate(hits, start=1):
            title = hit.get("title") or "Hacker News Discussion"
            story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
            points = hit.get("points", 0)
            author = hit.get("author", "anonymous")
            comments = hit.get("num_comments", 0)
            snippet = (
                f"{html.escape(title)} — {points} points by {html.escape(str(author))} "
                f"({comments} comments on Hacker News)"
            )
            results.append(
                SearchResult(
                    title=title,
                    url=story_url,
                    snippet=snippet,
                    source="Hacker News API",
                    source_rank=idx,
                    raw_score=float(points or 0),
                    metadata={"points": points, "comments": comments},
                )
            )
        return results
    except Exception:
        return []


def search_github_api(query: str, limit: int = 5) -> List[SearchResult]:
    """Queries the public GitHub REST Search API for open-source repositories."""
    params = urllib.parse.urlencode({"q": query, "sort": "stars", "order": "desc", "per_page": str(limit)})
    url = f"https://api.github.com/search/repositories?{params}"
    try:
        data = _fetch_json(url, headers={"Accept": "application/vnd.github+json"})
        items = data.get("items", [])
        results: List[SearchResult] = []
        for idx, repo in enumerate(items, start=1):
            full_name = repo.get("full_name", "repository")
            html_url = repo.get("html_url", "https://github.com")
            desc = html.escape(repo.get("description") or "No description provided.")
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language") or "Code"
            snippet = f"{desc} · ⭐ {stars:,} stars · Language: {html.escape(str(lang))}"
            results.append(
                SearchResult(
                    title=f"{full_name} ({lang})",
                    url=html_url,
                    snippet=snippet,
                    source="GitHub API",
                    source_rank=idx,
                    raw_score=float(stars),
                    metadata={"stars": stars, "language": lang},
                )
            )
        return results
    except Exception:
        return []


def search_azure_ai_search_api(query: str, limit: int = 6) -> List[SearchResult]:
    """
    Optional Microsoft Azure AI Search (Free Tier) REST API connector.
    Activates automatically when AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX,
    and AZURE_SEARCH_KEY are configured in environment variables.
    """
    if not (AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX and AZURE_SEARCH_KEY):
        return []

    url = (
        f"{AZURE_SEARCH_ENDPOINT}/indexes/{urllib.parse.quote(AZURE_SEARCH_INDEX)}"
        f"/docs/search?api-version={AZURE_API_VERSION}"
    )
    payload = {"search": query, "top": limit, "count": True}
    try:
        data = _fetch_json(url, headers={"api-key": AZURE_SEARCH_KEY}, post_body=payload)
        items = data.get("value", [])
        results: List[SearchResult] = []
        for idx, doc in enumerate(items, start=1):
            title = str(doc.get("title") or doc.get("name") or doc.get("id") or f"Azure Result #{idx}")
            doc_url = str(doc.get("url") or doc.get("metadata_storage_path") or "#")
            content = str(doc.get("content") or doc.get("description") or "")[:240]
            score = float(doc.get("@search.score", 0.0))
            results.append(
                SearchResult(
                    title=title,
                    url=doc_url,
                    snippet=html.escape(content),
                    source="Azure AI Search",
                    source_rank=idx,
                    raw_score=round(score, 4),
                )
            )
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF) Aggregator
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = RRF_K,
    source_weights: Optional[Dict[str, float]] = None,
) -> List[SearchResult]:
    """
    Combines ranked lists from multiple heterogeneous retrieval systems (Local BM25 + APIs)
    using Reciprocal Rank Fusion:
        RRF_Score(d) = sum_{s in Sources} weight(s) * (1 / (k + rank_s(d)))

    Deduplicates items by canonical URL and boosts documents found across multiple sources.
    """
    weights = source_weights or {
        "Local BM25 Index": 1.15,
        "Azure AI Search": 1.20,
        "Wikipedia API": 1.0,
        "Hacker News API": 0.95,
        "GitHub API": 0.95,
    }
    merged_by_url: Dict[str, SearchResult] = {}
    sources_by_url: Dict[str, List[str]] = {}

    for results in result_lists:
        for item in results:
            norm_url = item.url.rstrip("/").lower()
            w = weights.get(item.source, 1.0)
            incremental_rrf = w * (1.0 / (k + item.source_rank))

            if norm_url not in merged_by_url:
                item_copy = SearchResult(
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    source=item.source,
                    source_rank=item.source_rank,
                    raw_score=item.raw_score,
                    rrf_score=round(incremental_rrf, 6),
                    metadata=dict(item.metadata),
                )
                merged_by_url[norm_url] = item_copy
                sources_by_url[norm_url] = [item.source]
            else:
                existing = merged_by_url[norm_url]
                existing.rrf_score = round(existing.rrf_score + incremental_rrf, 6)
                if item.source not in sources_by_url[norm_url]:
                    sources_by_url[norm_url].append(item.source)
                    existing.source = " + ".join(sources_by_url[norm_url])

    ranked = sorted(merged_by_url.values(), key=lambda r: r.rrf_score, reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Unified Hybrid Search Orchestrator
# ---------------------------------------------------------------------------
class HybridPrivateSearchEngine:
    """Coordinates parallel API retrieval, local BM25 indexing, caching, and RRF ranking."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.local_index = LocalBM25Index(db_path=db_path)
        self.cache = PrivacyQueryCache(ttl_seconds=CACHE_TTL_SECONDS)

    def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        limit: int = 15,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        clean_query = query.strip()
        if not clean_query:
            return {"query": "", "results": [], "total": 0, "latency_ms": 0.0, "cached": False}

        active_sources = sources or ["local", "wikipedia", "hackernews", "github", "azure"]
        cache_key = f"{clean_query.lower()}::{'|'.join(sorted(active_sources))}::{limit}"

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached_copy = dict(cached)
                cached_copy["cached"] = True
                return cached_copy

        t0 = time.perf_counter()
        result_lists: List[List[SearchResult]] = []

        if "local" in active_sources:
            result_lists.append(self.local_index.search(clean_query, limit=8))

        tasks = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            if "wikipedia" in active_sources:
                tasks[pool.submit(search_wikipedia_api, clean_query, 6)] = "wikipedia"
            if "hackernews" in active_sources:
                tasks[pool.submit(search_hackernews_api, clean_query, 6)] = "hackernews"
            if "github" in active_sources:
                tasks[pool.submit(search_github_api, clean_query, 5)] = "github"
            if "azure" in active_sources and AZURE_SEARCH_ENDPOINT:
                tasks[pool.submit(search_azure_ai_search_api, clean_query, 6)] = "azure"

            for future in as_completed(tasks):
                try:
                    res = future.result()
                    if res:
                        result_lists.append(res)
                except Exception:
                    pass

        fused = reciprocal_rank_fusion(result_lists, k=RRF_K)[:limit]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        response_payload = {
            "query": clean_query,
            "total": len(fused),
            "latency_ms": latency_ms,
            "cached": False,
            "ranking_algorithm": f"Reciprocal Rank Fusion (k={RRF_K}) + Okapi BM25",
            "azure_ai_search_enabled": bool(AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY),
            "results": [asdict(item) for item in fused],
        }
        if use_cache:
            self.cache.set(cache_key, response_payload)
        return response_payload


# ---------------------------------------------------------------------------
# Embedded Privacy-First Web UI (HTML + CSS + Vanilla JS)
# ---------------------------------------------------------------------------
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="referrer" content="no-referrer" />
  <title>NexusSearch — Private Hybrid & Metasearch Engine</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --badge-bg: #0f172a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 1.25rem 2rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--accent);
    }
    .privacy-pill {
      font-size: 0.75rem;
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.35);
      padding: 0.25rem 0.65rem;
      border-radius: 999px;
      font-weight: 600;
    }
    main {
      max-width: 920px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem 1.25rem;
      flex: 1;
    }
    .search-box {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }
    input[type="text"] {
      flex: 1;
      padding: 0.85rem 1.15rem;
      border-radius: 0.5rem;
      border: 1px solid var(--border);
      background: var(--card);
      color: var(--text);
      font-size: 1rem;
      outline: none;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    button {
      padding: 0.85rem 1.35rem;
      border-radius: 0.5rem;
      border: none;
      background: var(--accent);
      color: #0f172a;
      font-weight: 700;
      cursor: pointer;
      font-size: 0.95rem;
    }
    button:hover { background: var(--accent-hover); color: #fff; }
    .filters {
      display: flex;
      flex-wrap: wrap;
      gap: 0.85rem;
      margin-bottom: 1.5rem;
      font-size: 0.85rem;
      color: var(--muted);
      align-items: center;
    }
    .filters label { display: flex; align-items: center; gap: 0.35rem; cursor: pointer; }
    .stats-bar {
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
    }
    .result-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.65rem;
      padding: 1.1rem 1.25rem;
      margin-bottom: 0.85rem;
    }
    .result-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 0.75rem;
      margin-bottom: 0.35rem;
    }
    .result-title {
      color: var(--accent);
      text-decoration: none;
      font-size: 1.08rem;
      font-weight: 600;
    }
    .result-title:hover { text-decoration: underline; }
    .badge {
      font-size: 0.72rem;
      padding: 0.2rem 0.55rem;
      border-radius: 0.35rem;
      background: var(--badge-bg);
      border: 1px solid var(--border);
      color: var(--muted);
      white-space: nowrap;
    }
    .result-url {
      font-size: 0.78rem;
      color: #64748b;
      margin-bottom: 0.45rem;
      word-break: break-all;
    }
    .result-snippet {
      font-size: 0.92rem;
      line-height: 1.5;
      color: #cbd5e1;
    }
    .result-snippet mark {
      background: rgba(56, 189, 248, 0.25);
      color: #fff;
      padding: 0 0.2rem;
      border-radius: 0.2rem;
    }
    .indexer-panel {
      margin-top: 2.25rem;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.65rem;
      padding: 1.25rem;
    }
    .indexer-panel h3 { font-size: 1rem; margin-bottom: 0.75rem; color: var(--accent); }
    .indexer-grid { display: grid; gap: 0.65rem; }
    textarea {
      width: 100%;
      padding: 0.75rem;
      border-radius: 0.5rem;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-family: inherit;
      resize: vertical;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>NexusSearch</span>
      <span class="privacy-pill">Zero-Log · BM25 + RRF Hybrid</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--muted);">
      <a href="/docs" style="color: var(--accent); text-decoration: none;">OpenAPI /docs</a>
    </div>
  </header>
  <main>
    <form class="search-box" onsubmit="runSearch(event)">
      <input type="text" id="q" placeholder="Search privately across Local BM25 Index, Wikipedia, Hacker News, GitHub & Azure..." value="azure python search" required />
      <button type="submit">Search</button>
    </form>
    <div class="filters">
      <span>Sources:</span>
      <label><input type="checkbox" class="src-cb" value="local" checked /> Local BM25 (SQLite FTS5)</label>
      <label><input type="checkbox" class="src-cb" value="wikipedia" checked /> Wikipedia API</label>
      <label><input type="checkbox" class="src-cb" value="hackernews" checked /> Hacker News API</label>
      <label><input type="checkbox" class="src-cb" value="github" checked /> GitHub API</label>
      <label><input type="checkbox" class="src-cb" value="azure" checked /> Azure AI Search</label>
    </div>
    <div id="stats" class="stats-bar">Ready to search without tracking cookies or IP logging.</div>
    <div id="results"></div>

    <div class="indexer-panel">
      <h3>Add Document to Local SQLite FTS5 (BM25) Index</h3>
      <form class="indexer-grid" onsubmit="indexCustomDoc(event)">
        <input type="text" id="doc-url" placeholder="Document URL (e.g. https://example.com/my-notes)" required />
        <input type="text" id="doc-title" placeholder="Document Title" required />
        <textarea id="doc-content" rows="3" placeholder="Document text content to index with Porter stemming & Okapi BM25..." required></textarea>
        <div><button type="submit">Index Document Locally</button> <span id="idx-msg" style="font-size:0.85rem;color:var(--muted);margin-left:0.5rem;"></span></div>
      </form>
    </div>
  </main>
  <script>
    async function runSearch(e) {
      if (e) e.preventDefault();
      const q = document.getElementById('q').value.trim();
      if (!q) return;
      const sources = Array.from(document.querySelectorAll('.src-cb:checked')).map(cb => cb.value).join(',');
      const statsEl = document.getElementById('stats');
      const resEl = document.getElementById('results');
      statsEl.textContent = 'Querying hybrid index & APIs...';
      resEl.innerHTML = '';
      try {
        const resp = await fetch(`/api/search?q=${encodeURIComponent(q)}&sources=${encodeURIComponent(sources)}`);
        const data = await resp.json();
        statsEl.textContent = `Found ${data.total} results in ${data.latency_ms} ms · ${data.ranking_algorithm} ${data.cached ? '(Cached)' : ''}`;
        if (!data.results || data.results.length === 0) {
          resEl.innerHTML = '<div class="result-card">No matching results found. Try indexing a custom document below!</div>';
          return;
        }
        resEl.innerHTML = data.results.map(r => `
          <div class="result-card">
            <div class="result-header">
              <a class="result-title" href="${r.url}" target="_blank" rel="noopener noreferrer">${r.title}</a>
              <span class="badge">${r.source} · RRF: ${r.rrf_score.toFixed(4)}</span>
            </div>
            <div class="result-url">${r.url}</div>
            <div class="result-snippet">${r.snippet}</div>
          </div>
        `).join('');
      } catch (err) {
        statsEl.textContent = 'Error executing search request.';
      }
    }

    async function indexCustomDoc(e) {
      e.preventDefault();
      const msgEl = document.getElementById('idx-msg');
      msgEl.textContent = 'Indexing...';
      const payload = {
        url: document.getElementById('doc-url').value,
        title: document.getElementById('doc-title').value,
        content: document.getElementById('doc-content').value,
        tags: 'custom'
      };
      const resp = await fetch('/api/index', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      if (resp.ok) {
        msgEl.textContent = 'Indexed into SQLite FTS5! Run a search above to test BM25 ranking.';
        document.getElementById('doc-url').value = '';
        document.getElementById('doc-title').value = '';
        document.getElementById('doc-content').value = '';
      } else {
        msgEl.textContent = 'Failed to index document.';
      }
    }

    window.addEventListener('DOMContentLoaded', () => runSearch());
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI Web Application & Endpoints
# ---------------------------------------------------------------------------
engine = HybridPrivateSearchEngine()

try:
    from fastapi import FastAPI, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    app = FastAPI(
        title="NexusSearch Private Hybrid Search Engine",
        description=(
            "Privacy-preserving metasearch and local SQLite FTS5 (BM25) search engine "
            "using Reciprocal Rank Fusion (RRF), ready for Microsoft Azure App Service."
        ),
        version="1.0.0",
    )

    class IndexDocumentRequest(BaseModel):
        url: str
        title: str
        content: str
        tags: str = "custom"

    class CrawlUrlRequest(BaseModel):
        url: str
        tags: str = "crawled"

    @app.middleware("http")
    async def privacy_headers_middleware(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/", response_class=HTMLResponse, summary="Search Engine Web Interface")
    def read_root() -> str:
        return INDEX_HTML

    @app.get("/api/search", summary="Hybrid BM25 + Metasearch API with Reciprocal Rank Fusion")
    def api_search(
        q: str = Query(..., min_length=1, description="Search query string"),
        sources: str = Query(
            "local,wikipedia,hackernews,github,azure",
            description="Comma-separated list of sources to query",
        ),
        limit: int = Query(15, ge=1, le=50, description="Maximum fused results to return"),
    ) -> JSONResponse:
        source_list = [s.strip().lower() for s in sources.split(",") if s.strip()]
        payload = engine.search(query=q, sources=source_list, limit=limit)
        return JSONResponse(content=payload)

    @app.post("/api/index", summary="Index a Custom Document into Local SQLite FTS5 (BM25)")
    def api_index_document(body: IndexDocumentRequest) -> JSONResponse:
        res = engine.local_index.index_document(
            url=body.url,
            title=body.title,
            content=body.content,
            tags=body.tags,
        )
        engine.cache.clear()
        return JSONResponse(content=res)

    @app.post("/api/crawl", summary="Crawl a Live URL and Index into Local SQLite FTS5")
    def api_crawl_url(body: CrawlUrlRequest) -> JSONResponse:
        res = engine.local_index.crawl_and_index_url(url=body.url, tags=body.tags)
        engine.cache.clear()
        return JSONResponse(content=res)

    @app.get("/api/health", summary="Health & Index Statistics Endpoint")
    def api_health() -> JSONResponse:
        return JSONResponse(
            content={
                "status": "healthy",
                "index": engine.local_index.stats(),
                "azure_ai_search_configured": bool(AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY),
            }
        )

except ImportError:
    app = None


# ---------------------------------------------------------------------------
# Built-in Zero-Dependency HTTP Server Fallback & CLI Self-Test
# ---------------------------------------------------------------------------
def run_stdlib_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Runs the search engine using Python's built-in http.server if FastAPI/Uvicorn aren't installed."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class SearchHTTPRequestHandler(BaseHTTPRequestHandler):
        def _send_headers(self, status: int = 200, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self._send_headers(200, "text/html; charset=utf-8")
                self.wfile.write(INDEX_HTML.encode("utf-8"))
            elif parsed.path == "/api/search":
                qs = urllib.parse.parse_qs(parsed.query)
                q = qs.get("q", [""])[0]
                sources_raw = qs.get("sources", ["local,wikipedia,hackernews,github,azure"])[0]
                source_list = [s.strip().lower() for s in sources_raw.split(",") if s.strip()]
                payload = engine.search(query=q, sources=source_list)
                self._send_headers(200, "application/json")
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            elif parsed.path == "/api/health":
                self._send_headers(200, "application/json")
                self.wfile.write(
                    json.dumps({"status": "healthy", "index": engine.local_index.stats()}).encode("utf-8")
                )
            else:
                self._send_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

        def do_POST(self) -> None:
            if self.path == "/api/index":
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                res = engine.local_index.index_document(
                    url=body.get("url", ""),
                    title=body.get("title", ""),
                    content=body.get("content", ""),
                    tags=body.get("tags", "custom"),
                )
                engine.cache.clear()
                self._send_headers(200, "application/json")
                self.wfile.write(json.dumps(res).encode("utf-8"))
            else:
                self._send_headers(404, "application/json")
                self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = HTTPServer((host, port), SearchHTTPRequestHandler)
    print(f"NexusSearch running privately on http://{host}:{port} (Zero-Log Mode)")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NexusSearch Private Hybrid Search Engine")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--self-test", action="store_true", help="Run offline BM25 + RRF verification test")
    args = parser.parse_args()

    if args.self_test:
        print("Running offline self-test (SQLite FTS5 BM25 + Reciprocal Rank Fusion)...")
        engine.local_index.index_document(
            url="https://azure.microsoft.com/en-us/pricing/free-services",
            title="Microsoft Azure Free Tier Services for Python Developers",
            content="Host Python FastAPI web apps on Azure App Service F1 Free Tier and connect to Azure AI Search.",
            tags="microsoft azure free",
        )
        test_res = engine.search("microsoft azure python", sources=["local"], use_cache=False)
        print(json.dumps(test_res, indent=2))
    else:
        try:
            import uvicorn

            if app is not None:
                uvicorn.run(app, host=args.host, port=args.port, access_log=False)
            else:
                run_stdlib_server(host=args.host, port=args.port)
        except ImportError:
            run_stdlib_server(host=args.host, port=args.port)