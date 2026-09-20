# Design Decisions

One-sentence decisions made while building the agent, as `TODO.md` asks for.

## Decision: `arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)`

**Decision:** The shared `arxiv.Client` used by `services/arxiv_client.py` is configured with `delay_seconds=3` and `num_retries=3`.

**Reason:** Respects arXiv's requested API politeness policy (no more than one request per 3 seconds) instead of hammering the endpoint on retries.

**Impact:** Topic searches and ID lookups are throttled client-side; a burst of searches will be visibly slower but won't risk a rate-limit ban.

## Decision: candidate ranking uses its own embedder, not `services/embeddings.py`

**Decision:** `services/ranker.py` loads its own lazily-cached `SentenceTransformer` instance rather than depending on the Part E embeddings service.

**Reason:** Part E (`services/embeddings.py`, chunk/QA embeddings) doesn't exist yet at the point ranking is needed, and ranking only ever embeds short title+abstract strings — a different, much smaller workload than paper-chunk embedding. Both currently default to the same `settings.embedding_model`, so behavior is consistent even though the code paths are separate.

**Impact:** Once Part E lands, `ranker.py`'s `_default_embed_fn` could be swapped to call `services/embeddings.py` directly to avoid loading two copies of the model; left as-is for now since it works and isn't required by any current test.

## Decision: `AgentState` gained `warnings`, `needs_ranking`, `selection_reason`, `selection_scores`

**Decision:** Added these four fields to `AgentState` (Phase 6 was already "done", but Phases 13–14 need them explicitly).

**Reason:** `TODO.md` Phase 13.4 and 14.6 name these fields directly (`state.warnings`, `needs_ranking`, `selection_reason`, `selection_scores`), and no existing field covers them — `errors` is reserved for failures, not soft warnings like a broadened search query.

**Impact:** Purely additive; existing state round-trip tests (`test_state.py`) still pass unchanged since all four have safe defaults.

## Decision: two new `Settings` fields — `arxiv_max_results`, `selection_confidence_threshold`

**Decision:** Added `arxiv_max_results: int = 20` (cap on arXiv topic-search candidates) and `selection_confidence_threshold: float = 0.35` (minimum embedding-similarity score to auto-select a paper).

**Reason:** `TODO.md` Phase 13.3 ("max_results from settings") and 14.5 ("confidence threshold") reference config values that didn't exist yet; `top_k`/`min_similarity` are reserved for the RAG/QA retrieval step (Part E/G), a different tuning knob than arXiv search-result count or paper-selection confidence.

**Impact:** Documented in `.env.example`. `top_k` remains QA-only; these two are search/selection-only.

## Decision: two-column ordering clusters on block `x0` (left edge), not x-midpoint

**Decision:** `services/pdf_parser._order_blocks_for_columns` clusters PyMuPDF text blocks by their left edge (`x0`) to detect a two-column layout, not by the x-midpoint `(x0+x1)/2` that `TODO.md` Phase 16.3 describes.

**Reason:** Midpoint clustering was tried first and produces false positives on ordinary single-column text: a short heading ("Abstract") sitting above a long paragraph has a much smaller midpoint than the paragraph's lines even though both share the same left edge, so the "bimodal" gap check misfired and reordered a single-column page (confirmed on the `tests/fixtures/sample_paper.pdf` fixture — the title/authors/abstract heading ended up after the abstract body). Left edges are nearly identical within one column regardless of line length, so clustering on `x0` only fires for genuine multi-column layouts (verified against `tests/fixtures/two_column_test.pdf`).

**Impact:** `services/pdf_parser.py` only. Single-column pages (the common case) are unaffected; real two-column papers still get column-aware ordering.

## Decision: font-size heading candidates are gathered in `extract_with_pymupdf`, not `detect_sections`

**Decision:** The Phase 18.2 font-size heuristic runs as part of `extract_with_pymupdf` (via `page.get_text("dict")`), returning candidate heading strings in the doc-metadata dict that `detect_sections` accepts as an optional `font_candidates` argument, rather than `detect_sections` re-opening the PDF itself.

**Reason:** `detect_sections(pages)`'s documented signature takes plain per-page text, not a PDF handle, but the font-size signal needs span-level data only available from PyMuPDF's own parse. Computing it during the extraction pass that's already open avoids reopening/reparsing the file a second time. The `pdfplumber` fallback path returns an empty `font_heading_candidates` list, since pdfplumber pages are only reached for `detect_sections` after `extract_with_pdfplumber` — the numbered/keyword heuristics still work, just without the font signal.

**Impact:** `services/pdf_parser.py`; `nodes/fetch_parse.py` threads `doc_meta["font_heading_candidates"]` through to `detect_sections`.

## Decision: `clean_text` runs twice in the fetch/parse pipeline

**Decision:** `nodes/fetch_parse.py` calls `pdf_parser.clean_text` once per page (right after extraction) and again on the joined body text right before storing `ParsedPaper.full_text`.

**Reason:** `clean_text`'s repeated-header/footer removal (Phase 16.5) only has something to detect once it sees a line recur several times, which is only visible once multiple pages' text sit in one string. The per-page pass handles the per-string-safe parts (ligatures, hyphenated line breaks, standalone page-number lines); the second pass on the joined text is where the header/footer dedup actually fires. Calling it on a single page is a harmless no-op for that part.

**Impact:** `services/pdf_parser.py`, `nodes/fetch_parse.py`.

## Decision: references and appendices are excluded from `ParsedPaper` before chunking

**Decision:** `services/pdf_parser.split_references` cuts the section list at the first References/Bibliography/Appendix(-ices) heading; `nodes/fetch_parse.py` builds `ParsedPaper.full_text`/`sections` only from the sections before that cut. The raw references are still kept in `ParsedPaper.references` for display, just not embedded.

**Reason:** Reference lists and appendices are mostly citation metadata/supplementary material with little topical signal; embedding them pollutes retrieval — a QA query is far more likely to spuriously match a reference-list author name or an appendix table than the actual answer (Phase 19.4's explicit rationale).

**Impact:** `services/chunker.py` (Part E, not built yet) will only ever see `ParsedPaper.sections` as already filtered — no separate exclusion logic needed there.

## Decision: new `Settings.max_pdf_size_mb` field (default 50)

**Decision:** Added `max_pdf_size_mb: int = 50`, checked by `download_pdf` against the streamed byte count so an oversized PDF is aborted mid-download rather than filling disk.

**Reason:** `TODO.md` Phase 15.5 asks for this cap "from settings"; no existing field covered it (`max_pdf_pages` limits post-extraction page count, not download size).

**Impact:** Documented in `.env.example`.

## Decision: `Chunk`/`RetrievedChunk` live in their service modules, not `state.py`

**Decision:** `Chunk` (Phase 20.1) is defined in `services/chunker.py`; `RetrievedChunk` (Phase 22.5) is defined in `services/vectorstore.py`. Neither was added to `state.py`.

**Reason:** `state.py` only holds models that are actual `AgentState` field types (`PaperMeta`, `Section`, `ParsedPaper`, `QATurn`). Chunks are an internal handoff between chunking and the vector store, never stored on `AgentState` directly — only their count (`chunk_count`) and the collection name are. This mirrors how `PaperMeta`/candidate-ranking types already work.

**Impact:** `nodes/chunk_embed.py` imports `Chunk` from `services.chunker`; `nodes/qa.py` (Part G, not built yet) will import `RetrievedChunk` from `services.vectorstore`.

## Decision: `chunk_section` builds chunk text as an exact slice of the source, not a re-join of its pieces

**Decision:** `services/chunker.chunk_section` tracks only `(start, end)` character offsets into `section.text` while packing pieces, and produces each chunk's `text` as `section.text[start:end].strip()` — never by `" ".join(pieces)`.

**Reason:** An early version joined paragraph/sentence/hard-split pieces with a literal `" "` separator. That's roughly right for sentences (which really were separated by whitespace) but wrong for hard-split character fragments, where inserting a space fabricates a character that was never in the source and silently pushed some chunks past the `size * 1.2` bound (caught by `tests/test_chunker.py`'s hard-split case: a 50-char piece + 10-char overlap + 1 fabricated space = 61 chars against a 60-char limit). Slicing the original text sidesteps the whole class of bug and makes `char_start`/`char_end` exact by construction instead of approximated.

**Impact:** `services/chunker.py` only. Max chunk length is now bounded by `size + overlap` (plus at most a couple of characters for a paragraph break inside that span), safely under `size * 1.2` for the default `chunk_overlap`/`chunk_size` ratio (150/1000 = 15%) — this ratio must stay ≤ ~20% for that bound to hold in general.

## Decision: Chroma collections use cosine distance; similarity recovered as `1 - distance`

**Decision:** `services/vectorstore.upsert_chunks` creates each paper's collection with `metadata={"hnsw:space": "cosine"}`. `search()` converts Chroma's returned distance to a similarity score via `score = 1.0 - distance`.

**Reason:** Chroma's default space is squared L2, which isn't directly a 0-1 similarity score. Cosine distance is `1 - cosine_similarity`, so with embeddings already L2-normalized (`services/embeddings.py`), `1 - distance` recovers the same cosine similarity the ranking step (`services/ranker.py`) and `settings.min_similarity`/`selection_confidence_threshold` already assume — one consistent similarity definition across the whole app. Verified empirically: an exact-match query embedding returns `distance = 0.0` → `score = 1.0`; an orthogonal one returns `distance = 1.0` → `score = 0.0`.

**Impact:** `services/vectorstore.py` only. `nodes/qa.py` (Part G) can apply `settings.min_similarity` directly against `RetrievedChunk.score` with no further conversion.

## Decision: `collection_exists(arxiv_id, expected_count=None)` — count check is optional, not a separate parameter name

**Decision:** `services/vectorstore.collection_exists` keeps the exact single-required-argument signature `TODO.md` Phase 22.4 specifies, adding `expected_count` as an optional second positional argument rather than a differently-named function.

**Reason:** The phase's prose ("skip re-embedding when the count matches") needs a count to compare against, but its own code signature only names `arxiv_id`. Making the count optional satisfies both: `collection_exists(id)` answers pure existence, `collection_exists(id, n)` answers the count-matching question `nodes/chunk_embed.py` actually needs.

**Impact:** `services/vectorstore.py`, `nodes/chunk_embed.py`.

## Decision: `tests/test_embeddings.py` uses the real sentence-transformers model; other Part E/C tests inject a fake `embed_fn`

**Decision:** Unlike `test_ranker.py`, `test_select_paper.py`, `test_vectorstore.py`, and `test_chunk_embed.py` (all of which inject a fake embedding function to stay network/model-free and fast), `test_embeddings.py` calls the real `embed_texts`/`embed_query` against the actual `all-MiniLM-L6-v2` model.

**Reason:** `services/embeddings.py` *is* the embedding wrapper being tested — faking the model out would leave the module itself unverified (dimension, normalization, determinism). `get_embedder()` is process-cached (`lru_cache`), so the ~55s model-load cost (mostly `sentence-transformers`/`torch` import + first construction in this environment, largely unaffected by `HF_HUB_OFFLINE` once the model is already cached locally) is paid once per test run, not once per test.

**Impact:** `tests/test_embeddings.py` is noticeably slower (~1 min) than the rest of the suite (~12s). Worth marking `network`/`slow` once Phase 39.6 sets up pytest markers, so it can be skipped in a fast local loop.

## Decision: `truncate_pages` returns original page indices alongside the kept pages

**Decision:** `services/pdf_parser.truncate_pages(pages, max_pages)` returns `(kept_pages, original_page_indices)` rather than just the truncated page list.

**Reason:** Once pages in the middle of a long paper are dropped (kept: first N + any later conclusion/results/discussion page), a downstream `Section.start_page` computed against the truncated list's own indices would report the wrong physical page number. `detect_sections` takes these original indices as `page_numbers` and maps offsets back through them.

**Impact:** `nodes/fetch_parse.py` threads `page_numbers` from `truncate_pages` into `detect_sections` so `Section.start_page` stays accurate even when Phase 17.4's truncation fires.

## Decision: summarization context budget is derived from `llm.max_context_tokens`, not a fixed constant

**Decision:** `nodes/summarize.budget_chars_for(llm)` computes the character budget for `build_context`/map-reduce from the configured LLM client's own `max_context_tokens` property (reserving 4000 tokens for prompt scaffolding + response, ~4 chars/token), falling back to a conservative 8000-token window when the client reports none (e.g. `NullLLM`).

**Reason:** `LLMClient.max_context_tokens` already exists (Phase 7.2) specifically "for prompt budgeting" per its docstring, and Gemini/Groq have very different real context windows (up to 1-2M tokens vs. Groq's much smaller free-tier models). A fixed constant would either waste Gemini's huge window or overflow a small Groq model.

**Impact:** `nodes/summarize.py` only. Switching `LLM_PROVIDER`/`LLM_MODEL` automatically changes how much of the paper gets included before map-reduce kicks in — no manual retuning needed.

## Decision: map-reduce trigger compares *raw* priority-section length to 2x the budget, not the whole paper

**Decision:** `summarize()` sums the character length of only the priority-selected sections (abstract/intro/method/results/conclusion) — not `parsed.full_text` — and switches to map-reduce when that sum exceeds `2 * budget_chars_for(llm)`.

**Reason:** `build_context`'s single-pass path only ever looks at those same priority sections anyway; comparing against the full paper (which also includes sections that get dropped either way) would trigger map-reduce for papers with long, irrelevant sections (e.g. a huge related-work section) that single-pass truncation already handles fine.

**Impact:** `nodes/summarize.py`. A paper with a very long Method or Results section relative to the LLM's context window is the realistic trigger case, exercised in `tests/test_summarize.py::test_summarize_uses_map_reduce_for_long_papers`.

## Decision: `nodes/qa.py` exposes both `answer_question` (returns sources) and `qa` (state-in/dict-out node wrapper)

**Decision:** Phase 29-31's QA logic lives behind two public entry points: `answer_question(state, question, ...) -> (QATurn, sources)` does the real work (retrieval, prompting, grounding/refusal, citation mapping) and returns the per-turn `sources` list (`{passage, section_title, chunk_index, score}` dicts) alongside the `QATurn`; `qa(state, question, ...) -> dict` is a thin wrapper matching every other node's `(state) -> partial-update dict` convention, returning only `{"messages": [...]}`.

**Reason:** `AgentState`/`QATurn` (Phase 6.5/31.1) have no field to hold "this turn's cited sources" - `QATurn.chunk_ids`/`scores` are the only per-turn record kept on state. Phase 37.3's CLI REPL ("render each answer in a panel with the cited sections listed underneath") needs the sources synchronously, right after asking, not by re-deriving them later. Returning a `sources` key from `qa()` itself would put a non-`AgentState`-schema key in the dict a future LangGraph node handler returns, which is risky once Part H (Phase 32, not built yet) actually registers `qa` as a graph node. Splitting the two avoids guessing Part H's exact node/state contract now while keeping both call shapes available.

**Impact:** `nodes/qa.py` only. Part H/I, when built, should have the CLI REPL call `answer_question` directly (it already needs `question` as an explicit argument since it isn't a state field - see the next decision) and reserve `qa()` for however the graph ends up re-entering at this node.

## Decision: `question` is an explicit argument to `retrieve`/`answer_question`/`qa`, not a new `AgentState` field

**Decision:** None of the three QA functions read the question from `AgentState`; all take it as an explicit parameter, exactly as `TODO.md` Phase 29.1 already specifies for `retrieve`.

**Reason:** Phase 32.5 states the QA loop is driven by the CLI re-entering the graph at the `qa` node rather than an internal cycle, but Part H (graph wiring) isn't built yet, so there's no established mechanism for how a per-turn question reaches a graph node. Adding a speculative field (e.g. `pending_question`) to `AgentState` now would be guessing at a Part H design decision that isn't mine to make in this pass.

**Impact:** `nodes/qa.py` only, for now. Whoever builds Part H picks how `question` reaches the `qa` node (a new `AgentState` field, or the CLI calling `answer_question` directly instead of going through the compiled graph for QA turns) - both remain equally possible with the current signatures.

## Decision: retrieval floor, dedup, and reorder all happen inside `retrieve`, in that order

**Decision:** `nodes/qa.retrieve` applies `settings.min_similarity` first, then deduplicates by normalized text prefix (keeping the higher-scoring chunk of a colliding pair), then sorts the survivors by `chunk_index` ascending.

**Reason:** `chunk_index` is already a flat, monotonically increasing index across the whole paper (assigned in document order by `chunker.chunk_paper`), so sorting by it is sufficient for "document position" with no extra bookkeeping. Filtering before deduping means a low-score near-duplicate can't survive by being compared only against other low-score chunks; deduping before sorting means the final order only has to reason about the (smaller) survivor set.

**Impact:** `nodes/qa.py` only. Dedup uses the first 120 normalized characters as the collision key - deliberately coarse, since it only needs to catch genuine overlap-region duplicates from adjacent chunks, not do general near-duplicate detection.

## Decision: the QA loop re-enters the graph via `update_state`, not a graph edge

**Decision:** `graph.py` registers `qa` as a node (so it shows in the diagram and satisfies Phase 32.2) but gives it no incoming edge - the compiled graph's only path is `understand_query -> ... -> summarize -> END`. A QA turn is computed off-graph by calling `nodes.qa.answer_question(state, question, ...)` directly, then persisted against the paper's existing thread with `compiled_graph.update_state(thread_config(arxiv_id), {"messages": [...]})`.

**Reason:** Phase 32.5 explicitly says the QA loop is "driven by the CLI re-entering at the `qa` node, not by an internal cycle" and asks for this decision to be written down. LangGraph's public API has no "invoke starting at an arbitrary node" operation - `graph.invoke()` always starts at `START`. The two real options were (a) add a `pending_question` field to `AgentState` and find some way to make a normal `invoke()` land on `qa`, which still can't target a single node without visiting everything before it, or (b) treat the checkpointed thread as the shared state and update it directly. `update_state` is exactly LangGraph's documented mechanism for injecting a value into a thread's state without running any node - verified directly (`tests/test_graph.py::test_update_state_appends_qa_turn_without_a_graph_edge`): it lands in the same SQLite-backed checkpoint `load_session` reads back. This keeps `question` as an explicit argument to `answer_question`/`qa` (already decided in Part G) instead of inventing a state field no other node needs.

**Impact:** `graph.py`; Part I's CLI REPL is expected to call `nodes.qa.answer_question` for each turn and `compiled_graph.update_state(...)` to persist it, rather than calling `compiled_graph.invoke(...)` for QA turns.

## Decision: `search_arxiv`'s zero-result case ends the graph directly, not through `handle_error`

**Decision:** `_route_after_search_arxiv` routes straight to `END` when neither `selected` nor `candidates` is set (the zero-results-after-broadening case), instead of Phase 33.2's literal "zero results → `handle_error`".

**Reason:** `search_arxiv` already sets `next_action = "ask_user_to_rephrase"` with the attempted queries recorded in `state.warnings` - a fully-specified "ask the user to rephrase" outcome, not a raw failure. This is the same shape of outcome as `select_paper`'s low-confidence case (Phase 33.3), which the plan itself already routes straight to `END` rather than through `handle_error`, precisely so the CLI can read `next_action` and drive an interactive prompt instead of seeing a generic error. Routing the zero-results case through `handle_error` instead would treat two identical "need user input" outcomes inconsistently for no benefit - `handle_error` doesn't have anything to add here that `next_action`/`warnings` don't already carry.

**Impact:** `graph.py` only. `handle_error` is reached only for the `UNKNOWN`-intent case and the `chunk_embed` embedding-failure case - both genuine failures with no existing follow-up context on state.

## Decision: `fetch_parse → chunk_embed` is a straight edge, not conditional

**Decision:** `graph.py` wires `fetch_parse → chunk_embed` unconditionally, instead of Phase 33.4's "parse OK → `chunk_embed`; total parse failure → `summarize` directly in abstract-only mode".

**Reason:** By the time Part H was built, `fetch_parse` (Phase 15-19) never actually fails outright - a download or parse failure already degrades to an abstract-only `ParsedPaper` and returns normally (never raises, never leaves `state.parsed` unset). Separately, `chunk_embed` (Phase 23.5, built before Part H) already has its own abstract-only handling: it detects `parse_method == "abstract_only"`, embeds a single abstract chunk, and sets `degraded_retrieval = True` - i.e., it already produces a queryable one-chunk collection for the degraded case. Skipping `chunk_embed` for that case, as the original phase 33.4 branch describes, would mean no vector collection ever gets created for an abstract-only paper, breaking the QA loop for exactly the papers that most need a graceful fallback. Always routing to `chunk_embed` lets its existing degrade logic do its job.

**Impact:** `graph.py` only. `chunk_embed`'s only conditional exit is the genuine failure case (`EmbeddingError`, wrapped by `_guarded_chunk_embed` and routed to `handle_error`) - Phase 33.5, unchanged from the plan.

## Decision: `chunk_embed`'s `EmbeddingError` is caught and turned into a state update, not left to propagate

**Decision:** `graph.py` wraps `nodes.chunk_embed.chunk_embed` in a local `_guarded_chunk_embed(state)` that catches `EmbeddingError` and returns `{"errors": [...], "next_action": "error"}` instead of letting the exception propagate; this wrapped version, not the raw node function, is what's registered as the `"chunk_embed"` graph node.

**Reason:** Phase 33.5 requires a conditional edge from `chunk_embed` on "embedding failure", but a LangGraph node that raises aborts the whole `invoke()` call before any conditional-edge function ever runs - there is no node to route from an exception. The only way to make "embedding failure → `handle_error`" a real graph edge is to have the node itself catch the failure and report it through the state it returns.

**Impact:** `graph.py` only; `nodes/chunk_embed.py` is unchanged and still raises `EmbeddingError` normally when called directly (e.g. in its own unit tests).

## Decision: `SqliteSaver` is constructed directly, not via its `from_conn_string` context manager

**Decision:** `graph.default_checkpointer(settings)` opens the SQLite connection itself (`sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)`) and passes it to `SqliteSaver(conn)` directly, rather than using `SqliteSaver.from_conn_string(...)` as a `with` block.

**Reason:** `from_conn_string` is a context manager whose connection closes when the `with` block exits - fine for a short script, but `build_graph()` needs the checkpointer to stay usable for the CLI's whole process lifetime (Part I, not built yet), well past the function call that creates it. Constructing the connection directly (with the same `check_same_thread=False` the context-manager helper itself uses) gives a `SqliteSaver` that lives as long as the process, closed implicitly at exit.

**Impact:** `graph.py` only. Confirms the `TODO.md` "uncertainty flag" about `langgraph-checkpoint-sqlite` living in a separate package: installed and pinned at `3.1.1` (compatible with `langgraph==1.2.11` / `langgraph-checkpoint==4.2.0`) in `pyproject.toml` and `requirements.txt`.

## Decision: `graph.run_pipeline_from_selection` reuses the same node functions `build_state_graph` wires, called directly rather than through the graph

**Decision:** The CLI resumes a session after `confirm_with_user` by calling a new `graph.run_pipeline_from_selection(state, ...)`, which runs `fetch_parse -> chunk_embed -> summarize` directly (with the same `EmbeddingError`-to-state-update guard `_guarded_chunk_embed` already uses), rather than trying to make the compiled graph itself resume mid-run.

**Reason:** The compiled graph has no interrupt to pause at — `select_paper`'s low-confidence branch already routes straight to `END` (Phase 33.3), so by the time the CLI needs to act there is nothing left for LangGraph to resume. This mirrors the QA loop's existing precedent (`update_state` instead of a graph edge, see the "QA loop re-enters via `update_state`" decision above): the CLI drives node functions directly and persists the result, rather than the graph.

**Impact:** `graph.py`, `cli.py`. Keeping this function in `graph.py` (not duplicated in `cli.py`) means the fetch_parse -> chunk_embed -> summarize ordering has one source of truth shared with `build_state_graph`'s own wiring.

## Decision: `summarize` is wrapped (`_safe_summarize`) so a missing LLM degrades instead of crashing the graph

**Decision:** `graph.py`'s `"summarize"` node is `_safe_summarize`, not the raw `nodes.summarize.summarize` — it catches `LLMError` and returns `{"warnings": [...]}` with `briefing` left unset, instead of letting the exception abort the run.

**Reason:** `chunk_embed` runs before `summarize`, so by the time summarization could fail for lack of an LLM key, parsing and retrieval have already succeeded — crashing the whole CLI at that point contradicts the assessment's "no paid API key required" constraint and `NullLLM`'s own documented intent ("parsing, retrieval, and extractive QA still work"). This gap was only found by actually running `digest` against a real paper with `LLM_PROVIDER=none` while verifying Part I.

**Impact:** `graph.py` (also used by `run_pipeline_from_selection`, so the confirm-and-resume path degrades the same way). `cli.py`'s `digest` command treats `briefing is None` (with `selected` set) as a degraded-but-successful run, not a hard failure.

## Decision: a rephrase retry gets a fresh session thread, not the one that just failed

**Decision:** `cli._run_digest`'s `ask_user_to_rephrase` loop computes a new `thread_id` for every attempt (recognizable arXiv IDs still get their own canonical thread), instead of re-invoking the graph on the same thread with the new topic.

**Reason:** A real bug, found while verifying the CLI: `search_arxiv`'s success path never resets `next_action` to `None`, so a successful retry on the *same* thread as a prior failed one inherited the stale `next_action="ask_user_to_rephrase"` from the checkpoint (LangGraph's per-key merge leaves an omitted key's checkpointed value untouched) and the CLI looped forever even after the graph had actually finished. This surfaced only under a fast, fully-mocked offline reproduction, not live testing — real arXiv/LLM latency between retries made the infinite loop look like ordinary network slowness.

**Impact:** `cli.py` only — `nodes/search_arxiv.py` (Part C) is unchanged. A fresh thread per rephrase attempt also means a failed attempt's `warnings`/`candidates` never leak into the next one, which is arguably the more correct behavior anyway.

## Decision: pronoun resolution rewrites the question text itself, not a separate "referent" field

**Decision:** `resolve_pronoun_reference` returns a single rewritten string (`"{question} - referring to: {previous_question}"`) that `retrieve` embeds directly, rather than embedding the raw question and the previous question separately and combining vectors.

**Reason:** Simpler, and avoids inventing a vector-combination scheme (e.g. averaging) that has no clear "right" weighting and isn't asked for by `TODO.md` Phase 31.4 ("rewrite it using the previous question before embedding" - text-level rewrite is the literal instruction).

**Impact:** `nodes/qa.py` only. The rewritten text is only used for embedding/retrieval; `QATurn.question` still stores the user's original, unmodified question.

## Fix: rate limiter wrapped outside retry, not inside, so retries evaded it

**Decision:** `llm/__init__.get_llm` now composes `base -> _RateLimitedClient -> _RetryingClient -> _CachedClient`, instead of `base -> _RetryingClient -> _RateLimitedClient -> _CachedClient`.

**Reason:** A real bug, found while capturing the Phase 44 live-demo transcript against Gemini's actual free tier: the old order acquired one rate-limit token per *logical* `complete()` call, then let `_RetryingClient` retry that same call up to 3 additional times against the real API without acquiring another token. A single summarize call that hit a transient `503` and retried still burned multiple real requests against the provider's per-minute quota while the local bucket only ever counted it as one — directly defeating Phase 8.3's "cannot exceed N requests/minute" guarantee. This surfaced immediately during the live run: the summarize call's retry burned enough of the real quota that the very next QA question got a `429 RESOURCE_EXHAUSTED` from Gemini.

**Impact:** `llm/__init__.py` only. Every real HTTP attempt, including retries, now goes through the token bucket first. Also revealed that Gemini's actual free-tier cap for `gemini-3.6-flash` is **5 requests/minute** (`quotaValue: '5'` in the live 429 response), not the ~15 RPM widely reported for the now-deprecated `gemini-2.0-flash` (Phase 42.6's default model, since renamed here — see the next entry). `Settings.llm_requests_per_minute` default and `.env`/`.env.example` were lowered from 15 to 5 to match.

## Fix: numbered-heading regex misdetected results-table rows as sections

**Issue:** `services/pdf_parser._numbered_heading_candidates` used `^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{0,100})$` with no bound on the number's plausibility. On a real paper (`1706.03762`), a results table extracted as flat text produced lines like `88.3 Petrov et al. (2006) [29]` and `23.75 Deep-Att + PosUnk [39]`, which match the regex just as well as a real heading like `3.2 Attention`. Each was kept as its own tiny "section," and `chunk_paper`'s `"[{section_title}] "` prefix turned them into short, citation-like chunks that scored anomalously high in cosine similarity for unrelated queries — found live: asking "What is the main contribution of this paper?" against the real `1706.03762` collection retrieved only these four junk chunks (scores 0.35-0.40) and refused with `NOT_IN_PAPER`, even though the real Abstract/Conclusion chunks scored lower (~0.20-0.25) and never made the top-k.

**Cause:** The regex has no way to distinguish a genuine section number (`3.2`, `6.1`) from a results-table score sharing a line with author/year/citation text (`88.3 Petrov et al. (2006) [29]`) or a bare year starting a wrapped sentence (`2014 English-French dataset ...`).

**Fix:** Added `_looks_like_section_number(number, title)`, called from `_numbered_heading_candidates`, rejecting a candidate when: the leading number component exceeds 20 (real papers don't have a "section 88"), any subsection component is zero-padded (`5.01` — real numbering never zero-pads), or the heading text matches a citation pattern (`et al.`, a `(YYYY)` year, or a trailing `[N]`).

**Impact:** `services/pdf_parser.py` only. Verified on the live `1706.03762` fixture: all 9 garbage sections disappear, "Results" section text is no longer fragmented, and retrieval for the same question now correctly surfaces Abstract/Method/Results chunks instead of citation fragments. New regression test: `tests/test_section_detection.py::test_detect_sections_ignores_results_table_rows_misread_as_headings`.

## Fix: an LLM failure during QA was indistinguishable from "not in the paper"

**Issue:** `nodes/qa.answer_question` caught `LLMError` (no key configured, rate limit, network failure) and returned the exact same refusal text as a genuine content-based `NOT_IN_PAPER` response ("I couldn't find that in this paper."). A user (or grader) hitting a rate limit or running with no API key would see a message implying the paper itself lacks the answer, when in fact the system never got an answer from the LLM at all — misleading in the same direction the assessment explicitly warns against (don't let the agent look confident/authoritative about something it doesn't actually know), even though the *cause* here is infrastructure, not hallucination.

**Cause:** The `except LLMError` branch reused `_REFUSAL_MESSAGE` instead of surfacing `exc.user_message`.

**Fix:** The except branch now returns `f"Could not get an answer right now: {exc.user_message}"` instead of the content-refusal string. `turn.grounded` stays `False` either way (correct — no grounded answer was produced), but the displayed text now honestly distinguishes "the paper doesn't say this" from "I couldn't reach the LLM to check."

**Impact:** `nodes/qa.py` only. Verified live: with `NullLLM` (`--provider none`) and a question whose chunks clear the similarity floor, the CLI now shows "Could not get an answer right now: No LLM API key is configured. ..." instead of a false "couldn't find that in this paper." New regression test: `tests/test_qa.py::test_llm_failure_refuses_with_an_honest_message_not_not_in_paper`.

## Fix: default `LLM_MODEL` (`gemini-2.0-flash`) was retired by Google

**Decision:** Changed the default `llm_model` (`config.py`, `.env.example`) from `gemini-2.0-flash` to `gemini-3.6-flash`, and added it to `llm/gemini.py`'s `_MAX_CONTEXT_TOKENS` map.

**Reason:** Found live while capturing the Phase 44 demo transcript: Gemini's API returned `404 NOT_FOUND` — `models/gemini-2.0-flash is no longer available`. `TODO.md`'s own "uncertainty flags" section anticipated exactly this class of drift for free-tier limits; model retirement is the same class of problem. `gemini-3.6-flash` is the current free-tier flash model as of 2026-09-19 (1M-token context, same RPD ballpark) per `ai.google.dev/gemini-api/docs/models`.

**Impact:** `config.py`, `.env.example`, `.env` (local, not committed), `llm/gemini.py`. Anyone re-running this later should expect Google to retire models again and should check `ai.google.dev/gemini-api/docs/models` if `digest` reports a 404 from Gemini.

## Fix: `build_graph`/`build_state_graph` never threaded the CLI's resolved `Settings` into graph nodes

**Decision:** `graph._default_nodes` now takes a `settings: Settings` argument and binds it into every node via `functools.partial` (`understand_query`, `search_arxiv`, `select_paper`, `fetch_parse`, `_guarded_chunk_embed`, `_safe_summarize`); `build_state_graph`/`build_graph` both now resolve and pass `settings` through instead of only using it to locate the checkpoint DB.

**Reason:** A real, significant bug, found while verifying Phase 47.3 ("confirm the app starts and gives a clear message with no API key set") for this README. `cli._resolve_settings` builds a `Settings` object with the CLI's `--provider`/`--top-k`/`--output-dir` overrides applied and passes it to `build_graph(settings=settings, ...)` — but `build_graph` only ever forwarded that object to `default_checkpointer(settings)`; `build_state_graph` built its node dict from the raw, unbound node functions, which LangGraph then calls as `node_fn(state)` with no extra arguments. Every node's own `settings = settings or get_settings()` fallback therefore always resolved to the process-wide, `.env`-only cached settings, silently ignoring any CLI override for the entire `digest` pipeline. Concretely: `arxiv-agent digest <id> --provider none` on a machine with a real `GEMINI_API_KEY` in `.env` still called the real Gemini API, because `summarize`'s node invocation never saw `llm_provider="none"`. (`run_pipeline_from_selection` and the QA REPL's `answer_question` calls were never affected — both call node functions directly with `settings=settings`, not through the compiled graph.)

**Impact:** `graph.py` only (plus two test stubs in `tests/test_graph.py` widened from `_raise(_state)` to `_raise(_state, **_kwargs)` to tolerate the now-present `settings` keyword). `--provider`/`--top-k`/`--output-dir` now actually take effect for the main `digest` command; verified live by re-running `digest --provider none` against both a never-before-seen paper (clean "no LLM API key configured" degraded output) and an already-cached one.

## Decision: `docs/graph.mmd`/`docs/graph.png` are hand-maintained, not auto-exported

**Decision:** Replaced the `export_graph_diagram`-generated `docs/graph.mmd` (plain `graph TD`, no edge labels, default LangGraph styling) with a hand-written Mermaid `flowchart TD` — labeled conditional edges, a legend distinguishing `add_edge` (solid) from `add_conditional_edges` (dotted), color-coded error/degraded nodes, and `qa` drawn in its own labeled subgraph reached by a dotted "resume from checkpoint" edge from `END` instead of floating with no incoming edge at all. Rendered to PNG with `mermaid-cli` (`npx @mermaid-js/mermaid-cli`, local render, no network call to mermaid.ink) at `-w 1150 -s 1.25` (≈1200px wide).

**Reason:** The auto-exported version is technically correct but illegible to a grader in the ~30 seconds it gets on screen in a 4-minute video: no edge labels means every branch looks equally likely, and an unlabeled disconnected `qa` node reads as a mistake rather than the deliberate "CLI re-enters via `update_state`, not a graph edge" design already documented elsewhere in this file.

**Impact:** `docs/graph.mmd` now carries a top comment (`Hand-maintained — if graph.py changes, update this file, it is no longer auto-generated`) and is the source of truth for `README.md`'s embedded Mermaid fallback block — both must be updated together if `graph.py`'s nodes or routing functions change. **Drift risk:** nothing enforces this sync automatically (`export_graph_diagram` still exists in `graph.py` and still works, but is no longer wired to produce the committed files) — a future change to `_route_after_*` functions or `_default_nodes` in `graph.py` will silently desync the diagram from the real graph unless whoever makes that change also re-edits `docs/graph.mmd` by hand. Verified line-by-line against every `add_edge`/`add_conditional_edges` call in `graph.py` as of 2026-09-19; see the full edge list and the `fetch_parse → chunk_embed` finding reported in that session's reply.

## Decision: main sequence clustered into a subgraph to stop long edges crossing unrelated nodes

**Decision:** In `docs/graph.mmd`, `handle_error` and `END` are declared *outside* a `pipeline` subgraph that contains `understand_query`/`search_arxiv`/`select_paper`/`fetch_parse`/`chunk_embed`/`summarize`. Also added a `%%{init: {"flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "curve": "linear"}}}%%` directive, and relabeled `understand_query → search_arxiv` from the vague "recognized intent" to **"paper lookup / topic search"**.

**Reason:** The first hand-maintained version (previous entry, same file) rendered correctly but was hard to trace at a glance: `understand_query → handle_error` ("unknown intent") and `chunk_embed`/`search_arxiv`/`select_paper` → `END` are edges between nodes at very different points in the sequence, so the layout engine routed them as long diagonals that swept across `fetch_parse`/`chunk_embed`/`select_paper`'s boxes on their way to the target — exactly the kind of line a viewer can't trace back to its source in a single video frame. Grouping the main sequence into its own compound node makes the layout route inter-cluster edges around the cluster's boundary rather than through its interior, which is what actually fixed the crossings (confirmed visually, not just theorized — see the "Important notes" below for what didn't work). The "recognized intent" label was also renamed because it obscured the paper-lookup-vs-topic-search distinction that's the first thing explained about this graph in the video reflection.

**Impact:** `docs/graph.mmd`, `docs/graph.png`, `README.md`'s embedded Mermaid fallback block. Rendered with `mermaid-cli` (`npx @mermaid-js/mermaid-cli`, local Puppeteer, no network dependency), plain `-w 1150 -b white` with **no `-s`/`--scale` flag** — see the next note for why.

**Important notes:**
1. **`understand_query → search_arxiv` is one edge, not two, and stays that way.** `_route_after_understand_query` in `graph.py` only returns `"handle_error"` or `"search_arxiv"` — both `PAPER_LOOKUP` and `TOPIC_SEARCH` intents route to the same node. Splitting this into two dotted edges (one per intent) would misrepresent the actual `StateGraph` topology, since LangGraph never branches here; the real ID-vs-topic behavior split happens one node downstream, inside `search_arxiv`'s own `_lookup`/`_topic_search` dispatch. The diagram's single edge is correct; only its label changed.
2. **`select_paper → END` ("low confidence") does not have a sibling "no candidates ranked" edge**, even though `select_paper`'s code has a second `next_action="ask_user_to_rephrase"` return path for an empty `embed_rank()` result. That path is unreachable through the compiled graph: `_route_after_search_arxiv` only ever routes to `select_paper` when `state.candidates` is non-empty, so `embed_rank` is never called with an empty list in practice. Drawing it as a second labeled edge would present dead code as a live, reachable branch — left undrawn deliberately, not missed.
3. **`mermaid-cli`'s `-s`/`--scale` flag produced a ghost artifact**: at `-s 1.25`, the rendered PNG showed a faint duplicate "Legend" label near the bottom of the image (outside the actual legend box), while the underlying SVG only ever contained one such text element (verified directly: `grep -c -i legend` on the intermediate `.svg` = 1). This reproduced consistently with `-s 1.25` and disappeared with `-s` omitted (default `1`), on a diagram tall enough to need Puppeteer to scroll/tile during capture — looks like a scale-vs-tiling interaction bug in this `mermaid-cli` version, not a diagram authoring error. Fix: never pass `-s`/`--scale` to `mmdc` for this diagram; render at native size (`-w 1150 -b white` gave 1134×1630px, within the "roughly 1000-1200px wide" target without any scale factor). If a future re-render needs a different size, verify the output for this artifact before committing it.

## Fix: no-LLM QA silently discarded retrieved chunks instead of using them

**Issue:** Found while auditing the "runs with zero API keys" claim for the video (see the prior session's report, same file's history). With `NullLLM` configured, `answer_question` still built the full QA prompt and called `llm.complete()`, which `NullLLM` always raises `LLMError` for — so a question whose chunks cleared `min_similarity` got a generic "Could not get an answer right now: No LLM API key is configured..." refusal despite retrieval having genuinely found relevant text. Separately, every refusal branch reached *after* chunks were retrieved (the LLM-failure branch, and the post-LLM `NOT_IN_PAPER` branch) always returned an empty `sources` list even though the retrieved chunks' IDs/scores were already being stored on the `QATurn` — so `/sources` showed "(no cited sources)" for a turn that had, in fact, retrieved something.

**Cause:** `answer_question` had no check for "no LLM configured" as distinct from "LLM configured but failing" — both hit the same `try: llm.complete() except LLMError` path. And both post-retrieval refusal branches (`except LLMError`, and the `NOT_IN_PAPER in response` check) hardcoded `return ..., []` instead of deriving the sources list from `chunks`, which was already in scope in both places.

**Fix:** `answer_question` now checks `llm.name == "none"` right after resolving the LLM and, if so, returns a new `_raw_passages_turn(question, chunks)` *without calling `complete()` at all* — the answer text is `_format_passages(chunks)` (the same `[N] (section) text` formatting already used to build the LLM prompt) under a `"No LLM configured — showing the most relevant passages from the paper."` header, with `grounded=True` and a full `sources` list. The pre-LLM guard (nothing clears `min_similarity`) is unchanged and still fires before this check, so an unrelated question still gets the ordinary refusal regardless of which LLM is configured. Separately, both the genuine-LLM-failure branch (`except LLMError`, e.g. rate limit, network, a real provider mid-outage) and the `NOT_IN_PAPER` branch now return `_sources_from_chunks(chunks)` instead of `[]`, so `/sources` reflects what was actually retrieved in every case where retrieval found something — refusal or not.

**Impact:** `nodes/qa.py` only (`_NO_LLM_HEADER`, `_sources_from_chunks`, `_raw_passages_turn`, and three call sites in `answer_question`); `llm/null.py`'s docstring and `README.md` updated to describe the new behavior. Three existing tests updated to match the corrected (not just differently-worded) behavior: `test_answer_question_with_null_llm_refuses_gracefully` → `test_answer_question_with_null_llm_returns_raw_passages` (was asserting the bug — empty sources, a refusal — as if it were correct), and both `test_unanswerable_question_refuses_via_not_in_paper_token`'s and `test_llm_failure_refuses_with_an_honest_message_not_not_in_paper`'s `sources == []` assertions updated to expect the real retrieved chunk. One new test added for the still-correct pre-LLM-refusal-with-NullLLM case. 172/172 tests pass, `ruff check`/`ruff format --check` clean. `graph.py` and every other node are unchanged — this was scoped to `nodes/qa.py` only, per explicit instruction.

## Known limitation (not fixed): a failed re-summarization on an already-digested paper still displays the old cached briefing

**Issue:** `_safe_summarize`'s failure branch returns `{"warnings": [...]}` without a `briefing` key. LangGraph's per-key checkpoint merge leaves an omitted key's previously-checkpointed value untouched (the same mechanism already documented above for the `next_action`/rephrase-loop bug). So: digest a paper once with a working LLM (real briefing gets checkpointed) → later re-run `digest` on that *same* paper ID with `--provider none` or a broken key → the CLI still displays and re-saves the **old, real briefing** from the checkpoint, with the "No LLM API key is configured" warning printed underneath it — technically true (this run made no LLM call) but easy to misread as "the briefing above was just produced with no key."

**Why not fixed here:** the two candidate fixes have a genuine tradeoff, not an obvious right answer. Resetting `briefing` to `None` whenever summarization fails would make `state.briefing` strictly reflect only the current run, but would also throw away a perfectly good previous briefing on a merely *transient* failure (e.g. a rate-limit blip on an otherwise-working key) — arguably worse UX than showing slightly-stale-but-correct content. The better fix (track whether the displayed briefing came from this run or an older checkpoint, and word the warning accordingly) needs a new state field and CLI-display change, which is more than a documentation-phase bug fix should take on without confirming the tradeoff. Left as a known, documented gap rather than guessed at.

**Impact:** `cli.py`'s `digest` command, only when re-running an *already-successfully-digested* paper with a different/broken provider. The first-time, never-digested-before path (the one the assessment's "no paid API key required" constraint is actually about) is unaffected and was verified clean (see the fix above).
