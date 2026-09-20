# Example run — vague topic search

Captured live against the Gemini free tier (`gemini-3.6-flash`) on 2026-09-19.

```
$ arxiv-agent digest "recent work on KV-cache compression for LLM inference" --no-qa
```

This run deliberately exercises two of the failure/edge cases the graph handles: the LLM-based query rewrite failed transiently (`503`, retried and gave up, falling back to the deterministic rewrite), and the first rewritten query returned **zero results**, triggering the one-shot broadened retry before a real search succeeded.

```
+---------------------------- arXiv Digest Agent -----------------------------+
| Provider: gemini                                                            |
| Embedding model: all-MiniLM-L6-v2                                           |
| Data directory: data                                                        |
+-----------------------------------------------------------------------------+
[19:30:18] INFO     AFC is enabled with max remote calls: 10.
[19:30:23] WARNING  Retrying gemini call in 1s (attempt 2/4) after: Gemini API
                    error (503): 503 UNAVAILABLE. (model experiencing high demand)
[19:30:24] INFO     AFC is enabled with max remote calls: 10.
[19:30:30] WARNING  Retrying gemini call in 2s (attempt 3/4) after: 503 UNAVAILABLE.
[19:30:32] INFO     AFC is enabled with max remote calls: 10.
[19:30:34] WARNING  Retrying gemini call in 4s (attempt 4/4) after: 503 UNAVAILABLE.
[19:30:38] INFO     AFC is enabled with max remote calls: 10.
[19:30:45] INFO     understand_query: LLM rewrite unavailable, used deterministic fallback
           INFO     Requesting page (first: True, try: 0):
                    https://export.arxiv.org/api/query?search_query=all%3A%22recent+work%22+AND+all%3A%22kv-cache+compression%22+AND+all%3A%22llm+inference%22...
[19:30:46] INFO     Got empty first page; stopping generation
           INFO     search_arxiv: zero results for 'all:"recent work" AND all:"kv-cache compression" AND all:"llm inference"',
                    retrying with 'all:recent work AND all:kv-cache compression'
           INFO     Sleeping: 2.997311 seconds
[19:30:49] INFO     Requesting page (first: True, try: 0):
                    https://export.arxiv.org/api/query?search_query=all%3Arecent+work+AND+all%3Akv-cache+compression...
           INFO     Got first page: 100 of 418827 total results
[19:31:03] WARNING  Warning: You are sending unauthenticated requests to the HF Hub. ...
Loading weights: 100%|##########| 103/103 [00:00<00:00, 1424.26it/s]
[19:31:12] WARNING  Retrying gemini call in 1s (attempt 2/4) after: Gemini API error (429):
                    429 RESOURCE_EXHAUSTED ... quotaValue: '5' ... (select_paper's LLM re-rank call)
[19:31:13] WARNING  Retrying gemini call in 2s (attempt 3/4) after: 429 RESOURCE_EXHAUSTED.
[19:31:16] WARNING  Retrying gemini call in 4s (attempt 4/4) after: 429 RESOURCE_EXHAUSTED.
[19:31:20] INFO     AFC is enabled with max remote calls: 10.
[19:31:28] INFO     select_paper: LLM ranking unavailable (Gemini API error (503): ...), using embedding rank
           INFO     download_pdf: cache hit for 2512.14946 at data\pdfs\2512.14946.pdf
[19:31:29] INFO     chunk_embed: 2512.14946 - collection already has 74 chunks, skipping re-embedding
[19:31:31] INFO     AFC is enabled with max remote calls: 10.
[19:31:37] WARNING  Retrying gemini call in 1s (attempt 2/4) after: 503 UNAVAILABLE.
[19:31:39] INFO     AFC is enabled with max remote calls: 10.
+---------------------------- Executive Briefing -----------------------------+
| EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM        |
| Serving                                                                     |
| Authors: Shaoting Feng, Yuhan Liu, Hanchen Li, Xiaokun Chen, Samuel Shen,   |
| Kuntai Du, Zhuohan Gu, Rui Zhang, Yuyang Huang, Yihua Cheng, Jiayi Yao,     |
| Qizheng Zhang, Ganesh Ananthanarayanan, Junchen Jiang                       |
| arXiv: 2512.14946  |  Published: 2025-12-16                                 |
| Link: http://arxiv.org/abs/2512.14946v1                                     |
+-----------------------------------------------------------------------------+
+----------------------------- Confidence Notes ------------------------------+
| Note that the text refers to the system as 'EVICPRESS' in most sections,    |
| but refers to it as 'CacheServe' in the conclusion section. The briefing    |
| treats them as referencing the same system described in the paper.         |
+-----------------------------------------------------------------------------+
+------------------------------ Why It Matters -------------------------------+
| Reusing key-value (KV) caches is crucial for fast LLM inference, but GPU    |
| memory fills up quickly as contexts and user counts grow. Existing         |
| approaches either evict caches to slower storage tiers using simple rules  |
| like LRU or compress caches in GPU memory independently, leading to        |
| suboptimal tradeoffs between latency and text quality. EVICPRESS jointly   |
| optimizes eviction and compression decisions across all contexts by        |
| recognizing that different contexts have varying sensitivities to          |
| compression errors. For software engineers building LLM serving           |
| infrastructure, this approach provides significantly faster                |
| time-to-first-token while maintaining output quality, allowing systems to  |
| serve more context efficiently on multi-tier storage hardware.             |
+-----------------------------------------------------------------------------+
Saved: data\outputs\2512.14946.md  data\outputs\2512.14946.json
Note: No results for 'all:"recent work" AND all:"kv-cache compression" AND
all:"llm inference"'; broadened to 'all:recent work AND all:kv-cache
compression'.
```

The full briefing (Method/Key Results/Limitations/Follow-up Questions tables, truncated above for length) is in [`examples/briefing_sample.json`](briefing_sample.json) and `data/outputs/2512.14946.md`.

**What this run demonstrates:**
- `understand_query`'s LLM rewrite failing transiently and falling back to the deterministic stopword-stripping rewrite (Phase 12.5) instead of blocking the run.
- `search_arxiv`'s zero-result broadened retry (Phase 13.4) — the first quoted, multi-term query matched nothing; dropping the most specific term found real candidates.
- `select_paper` falling back to pure embedding rank (Phase 14.4) when the LLM re-rank call also failed transiently, rather than crashing.
- `summarize` catching a genuine inconsistency in the source paper (it calls its own system two different names in different sections) and surfacing that honestly via `confidence_notes` instead of silently picking one.
- The retry/backoff wrapper (1s/2s/4s) absorbing repeated transient `503`/`429` errors from a real, rate-limited free-tier endpoint without the run failing outright.
