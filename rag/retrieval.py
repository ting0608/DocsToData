from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rag.models import Chunk, SearchHit
from rag.vector_store import FaissStore

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/CJK-character tokenizer used for BM25 keyword search.

    English: Kept intentionally simple (regex word split) to avoid pulling in
    a heavyweight NLP dependency for a small-corpus RAG project.
    中文: 刻意使用簡單的 regex 斷詞，避免為了小型語料庫的 RAG 專案引入過重的
    NLP 套件。
    """

    return _TOKEN_RE.findall(text.lower())


def keyword_search(
    store: FaissStore,
    query: str,
    top_k: int,
    source_filter: set[str] | None = None,
) -> list[SearchHit]:
    """BM25 keyword search over the same chunk corpus as the vector store.

    English: This is the "Keyword Search" half of the hybrid retrieval stage
    described in architecturev2.txt section 4. A fresh BM25 index is built
    from `store.chunks` each call; for the corpus sizes this project targets
    (single-digit to low-hundreds of documents) that is fast enough and
    avoids keeping a second persisted index in sync with FAISS.
    中文: 對應 architecturev2.txt 第 4 節混合檢索中的「關鍵字搜尋」部分。每次
    呼叫都會用 `store.chunks` 重新建立 BM25 索引；以本專案的語料規模（單一到
    數百份文件）來說速度已足夠，也不需要額外維護一份與 FAISS 同步的索引。
    """

    candidates = [
        (i, chunk) for i, chunk in enumerate(store.chunks) if not source_filter or chunk.source in source_filter
    ]
    if not candidates:
        return []

    corpus_tokens = [_tokenize(chunk.text) for _, chunk in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    hits: list[SearchHit] = []
    for (_, chunk), score in ranked[:top_k]:
        if score <= 0:
            continue
        hits.append(SearchHit(chunk=chunk, score=float(score)))
    return hits


def _chunk_key(chunk: Chunk) -> str:
    return chunk.chunk_id


def fuse_rrf(
    result_lists: list[list[SearchHit]],
    top_k: int,
    k: int = 60,
) -> list[SearchHit]:
    """Combine multiple ranked result lists with Reciprocal Rank Fusion (RRF).

    English: RRF is a simple, well-known way to merge rankings from
    heterogeneous retrievers (vector similarity and BM25 scores are not on
    the same scale, so summing raw scores would be meaningless). Each hit's
    fused score is `sum(1 / (k + rank))` across every list it appears in;
    `k=60` is the commonly used default from the original RRF paper/TREC
    usage. The returned score is the fused RRF score (not the original
    similarity), so callers should treat it as a ranking signal rather than a
    calibrated probability.
    中文: RRF 是合併多個來源排名結果的簡單常見做法（向量相似度與 BM25 分數
    不在同一個量尺上，直接加總原始分數沒有意義）。每個結果的融合分數為
    `sum(1 / (k + rank))`，加總其在每個列表中出現的名次；`k=60` 是 RRF 原始
    論文/TREC 慣用的預設值。回傳的分數是融合後的 RRF 分數（並非原始相似度），
    使用端應將其視為排序訊號，而非經過校準的機率值。
    """

    fused: dict[str, float] = {}
    chunk_by_key: dict[str, Chunk] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            key = _chunk_key(hit.chunk)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_key.setdefault(key, hit.chunk)

    ranked_keys = sorted(fused, key=lambda key: fused[key], reverse=True)
    return [SearchHit(chunk=chunk_by_key[key], score=fused[key]) for key in ranked_keys[:top_k]]


def rerank(query: str, candidates: list[SearchHit], top_k: int) -> list[SearchHit]:
    """Rerank a candidate set by lexical relevance to the query.

    English: This is a lightweight, dependency-free stand-in for a learned
    cross-encoder reranker (architecturev2.txt section 4, Stage 2). It scores
    each candidate by the fraction of query terms it contains plus a small
    bonus for exact substring matches, which is cheap and works reasonably
    well for factual PDF QA. Swap this function out for a real cross-encoder
    (e.g. `sentence-transformers` `CrossEncoder`) later without touching any
    caller — the signature (query, candidates, top_k) -> list[SearchHit] is
    the extension point.
    中文: 這是一個輕量、不依賴額外套件的 reranker，用來取代學習型
    cross-encoder（對應 architecturev2.txt 第 4 節 Stage 2）。做法是計算每個
    候選內容包含多少查詢詞彙的比例，並對完全符合的子字串給予小幅加分，成本低
    且對 PDF 事實型問答效果尚可。日後若要換成真正的 cross-encoder（例如
    `sentence-transformers` 的 `CrossEncoder`），只需替換這個函式本身即可，
    呼叫端完全不需修改——函式簽名 (query, candidates, top_k) -> list[SearchHit]
    就是預留的擴充點。
    """

    if not candidates:
        return []

    query_terms = set(_tokenize(query))
    if not query_terms:
        return candidates[:top_k]

    scored: list[tuple[float, SearchHit]] = []
    query_lower = query.lower()
    for hit in candidates:
        text_terms = set(_tokenize(hit.chunk.text))
        overlap = len(query_terms & text_terms) / len(query_terms)
        substring_bonus = 0.15 if query_lower in hit.chunk.text.lower() else 0.0
        # Blend in the upstream (vector/RRF) score as a tiebreaker so
        # candidates with zero lexical overlap still rank by prior relevance
        # rather than arbitrary list order.
        lexical_score = overlap + substring_bonus + (0.01 * hit.score)
        scored.append((lexical_score, hit))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    reranked = []
    for lexical_score, hit in scored[:top_k]:
        reranked.append(SearchHit(chunk=hit.chunk, score=lexical_score))
    return reranked


def hybrid_retrieve(
    store: FaissStore,
    query_text: str,
    query_vector: list[float],
    top_k: int = 5,
    candidate_k: int = 20,
    source_filter: set[str] | None = None,
) -> list[SearchHit]:
    """Full retrieval pipeline: vector + keyword candidates -> fuse -> rerank.

    English: Implements architecturev2.txt section 4 end to end:
    Stage 1 pulls `candidate_k` (default 20) results from vector similarity
    and BM25 keyword search each, Reciprocal-Rank-Fuses them into one
    candidate set, then Stage 2 reranks that set down to `top_k` (default 5)
    using `rerank()`.
    中文: 完整對應 architecturev2.txt 第 4 節的檢索流程：Stage 1 分別從向量
    相似度與 BM25 關鍵字搜尋各取出 `candidate_k`（預設 20）筆結果，以 RRF
    融合成一組候選集合，再由 Stage 2 使用 `rerank()` 縮減為 `top_k`（預設 5）
    筆最終結果。
    """

    vector_hits = store.search(query_vector, top_k=candidate_k, source_filter=source_filter)
    keyword_hits = keyword_search(store, query_text, top_k=candidate_k, source_filter=source_filter)

    candidates = fuse_rrf([vector_hits, keyword_hits], top_k=candidate_k)
    return rerank(query_text, candidates, top_k=top_k)
