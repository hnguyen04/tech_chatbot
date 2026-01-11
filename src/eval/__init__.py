"""
RAG Evaluation Module

This module provides comprehensive evaluation tools for RAG (Retrieval-Augmented Generation) systems.

Components:
- data_loader: Load and validate QA pairs from JSON
- lexical_metrics: BLEU, ROUGE-L, Token F1 (FREE)
- semantic_metrics: Embedding-based similarity (FREE)
- retrieval_metrics: Recall@K, MRR, Hit Rate, nDCG (FREE)
- llm_judge: LLM-as-Judge evaluation (PAID, budget-optimized)
- ablation: Ablation study utilities
- report_generator: Generate JSON and Markdown reports

Usage:
    # Run from command line (from project root)
    python -m src.eval.run_evaluation --qa-file data/qa_pairs.json
    
    # Or import and use programmatically
    from src.eval import LexicalEvaluator, SemanticEvaluator, RetrievalEvaluator
    
    lexical = LexicalEvaluator()
    results = lexical.evaluate_batch(predictions, references)
"""

# Data loading
from .data_loader import (
    QASample,
    EvalPrediction,
    load_qa_data,
    validate_qa_data,
    split_by_type,
    classify_query_type,
)

# Evaluators (FREE)
from .lexical_metrics import (
    LexicalEvaluator,
    tokenize_vietnamese,
    compute_bleu_1,
    compute_rouge_l,
    compute_token_f1,
)

from .semantic_metrics import (
    SemanticEvaluator,
    cosine_similarity,
    euclidean_distance,
)

from .retrieval_metrics import (
    RetrievalEvaluator,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
)

# LLM Judge (PAID)
from .llm_judge import (
    LLMJudge,
    JudgeResult,
    format_judge_results_for_report,
)

# Ablation study
from .ablation import (
    AblationRunner,
    AblationConfig,
    AblationResult,
    format_ablation_comparison,
    compute_ablation_delta,
)

# Report generation
from .report_generator import (
    EvaluationReport,
    save_json_report,
    save_markdown_report,
    save_full_report,
    generate_markdown_report,
    print_summary,
)

__all__ = [
    # Data loading
    "QASample",
    "EvalPrediction",
    "load_qa_data",
    "validate_qa_data",
    "split_by_type",
    "classify_query_type",
    
    # Lexical evaluation
    "LexicalEvaluator",
    "tokenize_vietnamese",
    "compute_bleu_1",
    "compute_rouge_l",
    "compute_token_f1",
    
    # Semantic evaluation
    "SemanticEvaluator",
    "cosine_similarity",
    "euclidean_distance",
    
    # Retrieval evaluation
    "RetrievalEvaluator",
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_hit_rate_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    
    # LLM Judge
    "LLMJudge",
    "JudgeResult",
    "format_judge_results_for_report",
    
    # Ablation
    "AblationRunner",
    "AblationConfig",
    "AblationResult",
    "format_ablation_comparison",
    "compute_ablation_delta",
    
    # Report
    "EvaluationReport",
    "save_json_report",
    "save_markdown_report",
    "save_full_report",
    "generate_markdown_report",
    "print_summary",
]
