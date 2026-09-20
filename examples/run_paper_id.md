# Example run — direct arXiv ID

Captured live against the Gemini free tier (`gemini-3.6-flash`) on 2026-09-19.

```
$ arxiv-agent digest 1706.03762
```

```
+---------------------------- arXiv Digest Agent -----------------------------+
| Provider: gemini                                                            |
| Embedding model: all-MiniLM-L6-v2                                           |
| Data directory: data                                                        |
+-----------------------------------------------------------------------------+
[19:34:46] INFO     Got first page: 1 of 1 total results
           INFO     download_pdf: cache hit for 1706.03762 at data\pdfs\1706.03762.pdf
[19:34:48] INFO     chunk_embed: 1706.03762 - collection already has 53 chunks, skipping re-embedding
[19:34:50] INFO     AFC is enabled with max remote calls: 10.
+---------------------------- Executive Briefing -----------------------------+
| Attention Is All You Need                                                   |
| Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion  |
| Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin                      |
| arXiv: 1706.03762  |  Published: 2017-06-12                                 |
| Link: http://arxiv.org/abs/1706.03762v7                                     |
+-----------------------------------------------------------------------------+
+------------------------------ Why It Matters -------------------------------+
| Traditional sequence processing models, like those used for language        |
| translation, process text sequentially token by token. This sequential      |
| nature prevents parallel execution across sequence lengths, creating a      |
| major computational bottleneck during training and scaling poorly to       |
| longer texts. This paper introduces the Transformer, a novel architecture   |
| that completely eliminates sequential recurrent layers in favor of          |
| self-attention mechanisms. By enabling full parallelization across input    |
| tokens, the Transformer significantly reduces training times - from weeks   |
| down to hours or days on standard GPU hardware - while achieving            |
| state-of-the-art accuracy in translation and demonstrating strong           |
| generalization to other language tasks.                                     |
+-----------------------------------------------------------------------------+
+----------------------------- Problem Statement -----------------------------+
| Dominant sequence models rely on recurrent neural networks that compute     |
| hidden states sequentially along symbol positions, precluding               |
| parallelization within training examples and creating severe computational  |
| bottlenecks. Existing attention mechanisms address distance dependencies    |
| but remain tied to underlying recurrent networks.                           |
+-----------------------------------------------------------------------------+
| Section             | Details                                               |
|---------------------+-------------------------------------------------------|
| Method              | - Replaces recurrent and convolutional layers in      |
|                      |   encoder-decoder architectures entirely with         |
|                      |   attention mechanisms.                               |
|                      | - Utilizes multi-headed self-attention to draw        |
|                      |   global dependencies without step-by-step            |
|                      |   sequential operations.                              |
|                      | - Enables full parallelization of sequence            |
|                      |   computation during training across all symbol       |
|                      |   positions.                                          |
| Key Results          | - BLEU 28.4 on WMT 2014 En-De, beating prior best      |
|                      |   (incl. ensembles) by over 2 BLEU.                   |
|                      | - New single-model SOTA BLEU 41.8 on WMT 2014 En-Fr    |
|                      |   after 3.5 days on 8 GPUs.                            |
|                      | - SOTA En-De reached after just 12 hours on 8 P100s.  |
|                      | - Generalized to English constituency parsing under    |
|                      |   both large and limited data regimes.                |
| Limitations          | - Inferred: handling very large inputs (images,        |
|                      |   audio, long video) unaddressed without local/        |
|                      |   restricted attention.                               |
|                      | - Inferred: inference-time generation stays            |
|                      |   sequential even though training is parallel.        |
|                      | - Inferred: evaluated on only two translation pairs    |
|                      |   and one non-translation task.                       |
| Follow-up Questions  | - How does memory scale with sequence length?         |
|                      | - What local/restricted attention would extend this   |
|                      |   to image/video?                                     |
|                      | - How to make inference-time generation less           |
|                      |   sequential?                                          |
|                      | - Performance on low-resource / non-Indo-European      |
|                      |   language pairs?                                     |
+-----------------------------------------------------------------------------+
Saved: data\outputs\1706.03762.md  data\outputs\1706.03762.json
+------------------------------------ QA -------------------------------------+
| Ask a question about the paper. Commands: /exit  /quit  /briefing  /sources |
| /history  /help                                                             |
+-----------------------------------------------------------------------------+
```

## QA exchange 1 — factual

```
> What optimizer and learning rate schedule did the authors use for training?

+---------------------------------- Answer -----------------------------------+
| The authors used the Adam optimizer with β1 = 0.9, β2 = 0.98, and           |
| ε = 10^-9 [4].                                                              |
|                                                                              |
| For the learning rate schedule, they varied the learning rate according to  |
| the formula:                                                                |
|   lrate = d_model^-0.5 · min(step_num^-0.5, step_num · warmup_steps^-1.5)   |
| This schedule increases the learning rate linearly for the first            |
| warmup_steps = 4000 training steps, and decreases it thereafter             |
| proportionally to the inverse square root of the step number [4].           |
|                                                                              |
| [4] 5.3 Optimizer (chunk 32, score 0.56)                                    |
+------------------------------------------------------------------------------+
```

Grounded, correctly cited to the actual "5.3 Optimizer" section with a real retrieval score.

## QA exchange 2 — requires synthesis across sections

```
> Why does the model need positional encodings given it has no recurrence or
  convolution, and why did the authors choose sinusoidal functions specifically?

+---------------------------------- Answer -----------------------------------+
| Because the model contains no recurrence and no convolution, positional     |
| encodings are necessary to inject information about the relative or        |
| absolute position of the tokens so that the model can make use of the       |
| sequence order [2].                                                         |
|                                                                              |
| The authors specifically chose sinusoidal functions for two reasons:        |
| 1. They hypothesized it would allow the model to easily learn to attend by  |
| relative positions, since for any fixed offset k, PE_(pos+k) can be         |
| represented as a linear function of PE_pos [3].                            |
| 2. The sinusoidal version may allow the model to extrapolate to sequence    |
| lengths longer than the ones encountered during training [3][4].            |
|                                                                              |
| [2] 3.5 Positional Encoding (chunk 22, score 0.69)                          |
| [3] 3.5 Positional Encoding (chunk 23, score 0.60)                          |
| [4] 3.5 Positional Encoding (chunk 24, score 0.48)                          |
+------------------------------------------------------------------------------+
```

Grounded, synthesizing two distinct reasons from three separate retrieved chunks within §3.5 (linked back to real chunk indices and similarity scores), rather than one flat lookup.

## QA exchange 3 — refusal (not in the paper)

```
> How does the Transformer compare to GPT-4 on reasoning benchmarks?

+---------------------------- Not Found in Paper -----------------------------+
| I couldn't find that in this paper. Closest sections checked: Abstract,    |
| Introduction, 5.4 Regularization, 6.1 Machine Translation, 6.2 Model       |
| Variations.                                                                |
+------------------------------------------------------------------------------+
```

The paper was published in 2017, years before GPT-4 existed — a clean refusal case. Note the retrieval floor still surfaced its closest (unrelated) sections for transparency rather than refusing silently.

_Updated 2026-09-19: the "closest sections checked" list above was regenerated after fixing a section-detection bug (see `docs/decisions.md`) — the original live capture showed `91.7 Transformer (4 layers)` / `92.1 Transformer (4 layers)`, which were results-table rows misdetected as section headings, not real sections. The refusal itself (a real, live `NOT_IN_PAPER` response) is unchanged; only the deterministic "closest sections" list, which is derived from retrieval metadata rather than the LLM call, was recomputed against the corrected chunk collection._

**What this run demonstrates:** a fully cached re-run (PDF and embeddings skip straight to the briefing call — Phase 20-23's "instant on re-run" design), a grounded factual answer with a real inline citation mapped back to its source section and similarity score, and a clean refusal on a question genuinely outside the paper's scope rather than a hallucinated comparison.

_A first attempt at the synthesis question above (about multi-head attention + positional encoding "working together") was itself refused — `top_k=5` retrieval didn't surface both relevant sections for that phrasing in the same query. That's an honest, real limitation of single-query-embedding retrieval at this `top_k`, not smoothed over here; see the README's "known limitations" and "what I'd do next" (hybrid BM25+dense retrieval, a reranker) for the fix. The second, more specific phrasing above was used instead to give a genuine synthesis example._
