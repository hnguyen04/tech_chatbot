"""
Ablation study utilities - Compare pipeline configurations.

Allows systematic comparison of different RAG configurations
to understand the contribution of each component.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time

from .data_loader import QASample, EvalPrediction
from .lexical_metrics import LexicalEvaluator
from .semantic_metrics import SemanticEvaluator
from .retrieval_metrics import RetrievalEvaluator


class AblationConfig(Enum):
    """Predefined ablation configurations."""
    BASELINE = "baseline"  # Full pipeline
    NO_RERANKING = "no_reranking"  # Skip reranking step
    NO_DECOMPOSITION = "no_decomposition"  # Skip query decomposition
    VECTOR_ONLY = "vector_only"  # Only vector search, no keyword
    AGENTIC = "agentic"  # Use AgenticRAG instead of Unified


@dataclass
class AblationResult:
    """Results from a single ablation configuration."""
    config_name: str
    predictions: List[EvalPrediction] = field(default_factory=list)
    lexical_metrics: Dict[str, float] = field(default_factory=dict)
    semantic_metrics: Dict[str, float] = field(default_factory=dict)
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    total_time_s: float = 0.0
    num_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "lexical_metrics": self.lexical_metrics,
            "semantic_metrics": self.semantic_metrics,
            "retrieval_metrics": self.retrieval_metrics,
            "avg_latency_ms": self.avg_latency_ms,
            "total_time_s": self.total_time_s,
            "num_samples": self.num_samples
        }


class AblationRunner:
    """
    Run ablation studies comparing different pipeline configurations.
    
    Configurations:
    - baseline: Full UnifiedRAG pipeline
    - no_reranking: Skip Qwen reranker
    - no_decomposition: Skip query decomposition (single query search)
    - agentic: Use AgenticRAG pipeline
    """
    
    def __init__(
        self,
        unified_pipeline=None,
        agentic_pipeline=None,
        run_semantic: bool = True
    ):
        """
        Initialize ablation runner.
        
        Args:
            unified_pipeline: UnifiedRAGPipeline instance
            agentic_pipeline: AgenticRAGPipeline instance (optional)
            run_semantic: Whether to run semantic evaluation (slower)
        """
        self.unified_pipeline = unified_pipeline
        self.agentic_pipeline = agentic_pipeline
        self.run_semantic = run_semantic
        
        # Initialize evaluators
        self.lexical_eval = LexicalEvaluator()
        self.retrieval_eval = RetrievalEvaluator()
        
        if run_semantic:
            self.semantic_eval = SemanticEvaluator()
        else:
            self.semantic_eval = None
    
    def _run_unified_baseline(
        self,
        samples: List[QASample]
    ) -> List[EvalPrediction]:
        """Run baseline UnifiedRAG."""
        predictions = []
        
        for sample in samples:
            result = self.unified_pipeline.query_for_eval(
                query=sample.question,
                skip_decomposition=False
            )
            
            pred = EvalPrediction(
                query=sample.question,
                ground_truth=sample.answer,
                prediction=result["answer"],
                context=result.get("context", ""),
                retrieved_ids=result.get("retrieved_ids", []),
                reranked_ids=result.get("reranked_ids", []),
                relevant_ids=sample.relevant_product_ids,
                sources=result.get("sources", [])
            )
            predictions.append(pred)
        
        return predictions
    
    def _run_no_decomposition(
        self,
        samples: List[QASample]
    ) -> List[EvalPrediction]:
        """Run without query decomposition."""
        predictions = []
        
        for sample in samples:
            result = self.unified_pipeline.query_for_eval(
                query=sample.question,
                skip_decomposition=True  # Skip decomposition
            )
            
            pred = EvalPrediction(
                query=sample.question,
                ground_truth=sample.answer,
                prediction=result["answer"],
                context=result.get("context", ""),
                retrieved_ids=result.get("retrieved_ids", []),
                reranked_ids=result.get("reranked_ids", []),
                relevant_ids=sample.relevant_product_ids,
                sources=result.get("sources", [])
            )
            predictions.append(pred)
        
        return predictions
    
    def _run_no_reranking(
        self,
        samples: List[QASample]
    ) -> List[EvalPrediction]:
        """Run without reranking step (use retrieved order directly)."""
        predictions = []
        
        for sample in samples:
            # Use query_for_eval but skip reranking by setting top_n = top_k
            # This effectively uses the retrieved order without reranking
            result = self.unified_pipeline.query_for_eval(
                query=sample.question,
                skip_decomposition=False,
                top_n=self.unified_pipeline.retriever.default_k if hasattr(self.unified_pipeline.retriever, 'default_k') else 20  # Large top_n to skip effective reranking
            )
            
            pred = EvalPrediction(
                query=sample.question,
                ground_truth=sample.answer,
                prediction=result["answer"],
                context=result.get("context", ""),
                retrieved_ids=result.get("retrieved_ids", []),
                reranked_ids=result.get("retrieved_ids", []),  # Same as retrieved (no reranking)
                relevant_ids=sample.relevant_product_ids,
                sources=result.get("sources", [])
            )
            predictions.append(pred)
        
        return predictions
    
    def _run_agentic(
        self,
        samples: List[QASample]
    ) -> List[EvalPrediction]:
        """Run AgenticRAG pipeline."""
        if not self.agentic_pipeline:
            raise ValueError("AgenticRAG pipeline not provided")
        
        predictions = []
        
        for sample in samples:
            result = self.agentic_pipeline.query_for_eval(
                user_query=sample.question
            )
            
            pred = EvalPrediction(
                query=sample.question,
                ground_truth=sample.answer,
                prediction=result["answer"],
                context=result.get("context", ""),
                retrieved_ids=result.get("retrieved_ids", []),
                reranked_ids=result.get("reranked_ids", []),
                relevant_ids=sample.relevant_product_ids,
                sources=result.get("sources", []),
                tool_log=result.get("tool_log", []),
                iterations=result.get("iterations", 0)
            )
            predictions.append(pred)
        
        return predictions
    
    def _evaluate_predictions(
        self,
        predictions: List[EvalPrediction]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a set of predictions."""
        results = {}
        
        # Extract texts
        pred_texts = [p.prediction for p in predictions]
        ref_texts = [p.ground_truth for p in predictions]
        
        # Lexical metrics
        results["lexical"] = self.lexical_eval.evaluate_batch(pred_texts, ref_texts)
        
        # Semantic metrics (if enabled)
        if self.semantic_eval and self.run_semantic:
            results["semantic"] = self.semantic_eval.evaluate_batch(pred_texts, ref_texts)
        
        # Retrieval metrics (only if we have relevant IDs)
        has_relevant = any(p.relevant_ids for p in predictions)
        if has_relevant:
            all_retrieved = [p.retrieved_ids for p in predictions]
            all_relevant = [p.relevant_ids for p in predictions]
            results["retrieval"] = self.retrieval_eval.evaluate_batch(
                all_retrieved, all_relevant
            )
        
        return results
    
    def run_single_config(
        self,
        config: AblationConfig,
        samples: List[QASample],
        verbose: bool = True
    ) -> AblationResult:
        """
        Run a single ablation configuration.
        
        Args:
            config: Configuration to run
            samples: QA samples to evaluate
            verbose: Print progress
            
        Returns:
            AblationResult with metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running configuration: {config.value}")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run appropriate configuration
        if config == AblationConfig.BASELINE:
            predictions = self._run_unified_baseline(samples)
        elif config == AblationConfig.NO_DECOMPOSITION:
            predictions = self._run_no_decomposition(samples)
        elif config == AblationConfig.NO_RERANKING:
            predictions = self._run_no_reranking(samples)
        elif config == AblationConfig.AGENTIC:
            predictions = self._run_agentic(samples)
        else:
            raise ValueError(f"Unsupported config: {config}")
        
        total_time = time.time() - start_time
        avg_latency = (total_time * 1000) / len(samples) if samples else 0
        
        # Evaluate
        if verbose:
            print(f"Evaluating {len(predictions)} predictions...")
        
        metrics = self._evaluate_predictions(predictions)
        
        result = AblationResult(
            config_name=config.value,
            predictions=predictions,
            lexical_metrics=metrics.get("lexical", {}),
            semantic_metrics=metrics.get("semantic", {}),
            retrieval_metrics=metrics.get("retrieval", {}),
            avg_latency_ms=avg_latency,
            total_time_s=total_time,
            num_samples=len(samples)
        )
        
        if verbose:
            print(f"Completed in {total_time:.1f}s (avg {avg_latency:.0f}ms/query)")
        
        return result
    
    def run_ablation_study(
        self,
        samples: List[QASample],
        configs: Optional[List[AblationConfig]] = None,
        verbose: bool = True
    ) -> Dict[str, AblationResult]:
        """
        Run full ablation study with multiple configurations.
        
        Args:
            samples: QA samples to evaluate
            configs: Configurations to test (default: baseline + no_decomposition)
            verbose: Print progress
            
        Returns:
            Dictionary mapping config name to results
        """
        if configs is None:
            configs = [AblationConfig.BASELINE, AblationConfig.NO_DECOMPOSITION]
            if self.agentic_pipeline:
                configs.append(AblationConfig.AGENTIC)
        
        results = {}
        
        for config in configs:
            try:
                result = self.run_single_config(config, samples, verbose)
                results[config.value] = result
            except Exception as e:
                print(f"Error running {config.value}: {e}")
                continue
        
        return results


def format_ablation_comparison(results: Dict[str, AblationResult]) -> str:
    """
    Format ablation results as a comparison table.
    
    Args:
        results: Dictionary of ablation results
        
    Returns:
        Markdown formatted comparison table
    """
    if not results:
        return "No results to display"
    
    lines = [
        "## Ablation Study Results\n",
        "### Comparison Table\n",
    ]
    
    # Determine which metrics to show
    configs = list(results.keys())
    
    # Lexical metrics table
    lines.append("#### Lexical Metrics\n")
    lines.append("| Configuration | BLEU-1 | ROUGE-L F1 | Token F1 |")
    lines.append("|---------------|--------|------------|----------|")
    
    for config_name, result in results.items():
        lex = result.lexical_metrics
        bleu = lex.get("bleu_1_mean", 0)
        rouge = lex.get("rouge_l_f1_mean", 0)
        f1 = lex.get("token_f1_mean", 0)
        lines.append(f"| {config_name} | {bleu:.4f} | {rouge:.4f} | {f1:.4f} |")
    
    # Semantic metrics table (if available)
    has_semantic = any(r.semantic_metrics for r in results.values())
    if has_semantic:
        lines.append("\n#### Semantic Metrics\n")
        lines.append("| Configuration | Similarity Mean | Similarity Std |")
        lines.append("|---------------|-----------------|----------------|")
        
        for config_name, result in results.items():
            sem = result.semantic_metrics
            mean = sem.get("semantic_similarity_mean", 0)
            std = sem.get("semantic_similarity_std", 0)
            lines.append(f"| {config_name} | {mean:.4f} | {std:.4f} |")
    
    # Retrieval metrics table (if available)
    has_retrieval = any(r.retrieval_metrics for r in results.values())
    if has_retrieval:
        lines.append("\n#### Retrieval Metrics\n")
        lines.append("| Configuration | Recall@10 | Precision@10 | MRR | Hit Rate |")
        lines.append("|---------------|-----------|--------------|-----|----------|")
        
        for config_name, result in results.items():
            ret = result.retrieval_metrics
            recall = ret.get("recall@10_mean", 0)
            prec = ret.get("precision@10_mean", 0)
            mrr = ret.get("mrr_mean", 0)
            hit = ret.get("hit_rate@10_mean", 0)
            lines.append(f"| {config_name} | {recall:.4f} | {prec:.4f} | {mrr:.4f} | {hit:.4f} |")
    
    # Latency table
    lines.append("\n#### Latency\n")
    lines.append("| Configuration | Avg Latency (ms) | Total Time (s) |")
    lines.append("|---------------|------------------|----------------|")
    
    for config_name, result in results.items():
        lines.append(f"| {config_name} | {result.avg_latency_ms:.0f} | {result.total_time_s:.1f} |")
    
    return "\n".join(lines)


def compute_ablation_delta(
    baseline: AblationResult,
    ablation: AblationResult
) -> Dict[str, float]:
    """
    Compute the difference (delta) between baseline and ablation.
    
    Positive delta means ablation is better.
    
    Args:
        baseline: Baseline configuration results
        ablation: Ablation configuration results
        
    Returns:
        Dictionary of metric deltas
    """
    deltas = {}
    
    # Lexical metrics (higher is better)
    for key in ["bleu_1_mean", "rouge_l_f1_mean", "token_f1_mean"]:
        baseline_val = baseline.lexical_metrics.get(key, 0)
        ablation_val = ablation.lexical_metrics.get(key, 0)
        deltas[f"delta_{key}"] = ablation_val - baseline_val
    
    # Semantic metrics (higher is better)
    if baseline.semantic_metrics and ablation.semantic_metrics:
        baseline_sim = baseline.semantic_metrics.get("semantic_similarity_mean", 0)
        ablation_sim = ablation.semantic_metrics.get("semantic_similarity_mean", 0)
        deltas["delta_semantic_similarity"] = ablation_sim - baseline_sim
    
    # Retrieval metrics (higher is better)
    if baseline.retrieval_metrics and ablation.retrieval_metrics:
        for key in ["recall@10_mean", "mrr_mean", "hit_rate@10_mean"]:
            baseline_val = baseline.retrieval_metrics.get(key, 0)
            ablation_val = ablation.retrieval_metrics.get(key, 0)
            deltas[f"delta_{key}"] = ablation_val - baseline_val
    
    # Latency (lower is better, so negate)
    deltas["delta_latency_ms"] = baseline.avg_latency_ms - ablation.avg_latency_ms
    
    return deltas


# Example usage
if __name__ == "__main__":
    print("Ablation study module loaded")
    print("Use AblationRunner to run ablation studies")
