"""Retrieval evaluation toolkit.

Provides ranking metrics, an eval-set loader, a runner that scores a
retriever against labeled queries, and the ``auto-rag-eval`` CLI. Used to
measure and regress-test retrieval quality across releases.

* :mod:`auto_rag.eval.metrics` - precision@k, recall@k, hit@k, MRR@k, nDCG@k.
* :mod:`auto_rag.eval.loader`  - labeled eval-set loading and validation.
* :mod:`auto_rag.eval.runner`  - scoring a retriever against an eval set.
"""
