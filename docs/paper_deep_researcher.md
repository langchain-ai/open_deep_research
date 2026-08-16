# PaperDeepResearcher

PaperDeepResearcher is a fork of LangChain Open Deep Research adapted for computer science literature review.

The first version adds:

- arXiv and Semantic Scholar paper retrieval
- metadata normalization and deduplication
- explainable relevance reranking
- PDF evidence extraction
- Chinese literature review output grounded in evidence snippets

The first version does not add a new frontend. It focuses on research quality and traceability.

## MVP Acceptance Checks

- Submit a CS research question.
- Generate 5 to 8 English search queries.
- Retrieve and normalize paper records from academic sources.
- Rank top 10 candidates with visible reasons.
- Deep-read top 3 PDFs when available.
- Generate a Chinese report with required sections.
- Attach evidence excerpts to deep-read claims.
- Mark missing venue metadata as unknown.

## Suggested Resume Bullet

Forked and extended LangChain's Open Deep Research into a CS literature research agent, adding arXiv and Semantic Scholar paper retrieval, metadata normalization, relevance reranking, PDF evidence extraction, and Chinese citation-grounded literature review generation.
