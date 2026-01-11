"""
Report generator for evaluation results.

Generates:
- JSON file with raw metrics data
- Markdown report with formatted tables and analysis
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .ablation import AblationResult, format_ablation_comparison
from .llm_judge import format_judge_results_for_report


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    pipeline_name: str
    num_samples: int
    lexical_metrics: Dict[str, float]
    semantic_metrics: Dict[str, float]
    retrieval_metrics: Dict[str, float]
    llm_judge_metrics: Optional[Dict[str, Any]] = None
    ablation_results: Optional[Dict[str, Any]] = None
    agentic_metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "pipeline_name": self.pipeline_name,
            "num_samples": self.num_samples,
            "metrics": {
                "lexical": self.lexical_metrics,
                "semantic": self.semantic_metrics,
                "retrieval": self.retrieval_metrics,
                "llm_judge": self.llm_judge_metrics,
                "agentic": self.agentic_metrics
            },
            "ablation": self.ablation_results,
            "config": self.config
        }


def save_json_report(
    results: Dict[str, Any],
    output_path: str = "eval_results.json"
) -> str:
    """
    Save evaluation results as JSON.
    
    Args:
        results: Evaluation results dictionary
        output_path: Output file path
        
    Returns:
        Path to saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"JSON report saved to: {path}")
    return str(path)


def generate_markdown_report(
    results: Dict[str, Any],
    pipeline_name: str = "RAG Pipeline",
    include_details: bool = True
) -> str:
    """
    Generate markdown report from evaluation results.
    
    Args:
        results: Evaluation results dictionary
        pipeline_name: Name of the pipeline being evaluated
        include_details: Include detailed per-sample results
        
    Returns:
        Markdown formatted report string
    """
    lines = []
    
    # Header
    lines.append(f"# Evaluation Report: {pipeline_name}")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Samples Evaluated**: {results.get('num_samples', 'N/A')}")
    lines.append("")
    
    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("1. [Summary](#summary)")
    lines.append("2. [Lexical Metrics](#lexical-metrics)")
    lines.append("3. [Semantic Metrics](#semantic-metrics)")
    lines.append("4. [Retrieval Metrics](#retrieval-metrics)")
    if results.get("llm_judge"):
        lines.append("5. [LLM-as-Judge](#llm-as-judge)")
    if results.get("ablation"):
        lines.append("6. [Ablation Study](#ablation-study)")
    if results.get("agentic"):
        lines.append("7. [Agentic Analysis](#agentic-analysis)")
    lines.append("")
    
    # Summary Section
    lines.append("## Summary")
    lines.append("")
    lines.append("### Key Metrics Overview")
    lines.append("")
    lines.append("| Category | Metric | Value |")
    lines.append("|----------|--------|-------|")
    
    # Add key metrics to summary
    lexical = results.get("lexical", {})
    semantic = results.get("semantic", {})
    retrieval = results.get("retrieval", {})
    llm_judge = results.get("llm_judge", {})
    
    if lexical:
        lines.append(f"| Lexical | BLEU-1 | {lexical.get('bleu_1_mean', 0):.4f} |")
        lines.append(f"| Lexical | ROUGE-L F1 | {lexical.get('rouge_l_f1_mean', 0):.4f} |")
        lines.append(f"| Lexical | Token F1 | {lexical.get('token_f1_mean', 0):.4f} |")
    
    if semantic:
        lines.append(f"| Semantic | Similarity | {semantic.get('semantic_similarity_mean', 0):.4f} |")
    
    if retrieval:
        lines.append(f"| Retrieval | Recall@10 | {retrieval.get('recall@10_mean', 0):.4f} |")
        lines.append(f"| Retrieval | MRR | {retrieval.get('mrr_mean', 0):.4f} |")
    
    if llm_judge and "faithfulness" in llm_judge:
        lines.append(f"| LLM Judge | Faithfulness | {llm_judge['faithfulness'].get('mean', 0):.2f}/5 |")
        lines.append(f"| LLM Judge | Relevance | {llm_judge['relevance'].get('mean', 0):.2f}/5 |")
        lines.append(f"| LLM Judge | Completeness | {llm_judge['completeness'].get('mean', 0):.2f}/5 |")
    
    lines.append("")
    
    # Lexical Metrics Section
    lines.append("## Lexical Metrics")
    lines.append("")
    lines.append("Lexical metrics measure word-level overlap between generated answers and ground truth.")
    lines.append("")
    
    if lexical:
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        
        metric_pairs = [
            ("BLEU-1", "bleu_1"),
            ("ROUGE-L Precision", "rouge_l_precision"),
            ("ROUGE-L Recall", "rouge_l_recall"),
            ("ROUGE-L F1", "rouge_l_f1"),
            ("Token Precision", "token_precision"),
            ("Token Recall", "token_recall"),
            ("Token F1", "token_f1"),
        ]
        
        for label, key in metric_pairs:
            mean = lexical.get(f"{key}_mean", 0)
            std = lexical.get(f"{key}_std", 0)
            lines.append(f"| {label} | {mean:.4f} | {std:.4f} |")
    else:
        lines.append("*No lexical metrics available*")
    
    lines.append("")
    
    # Semantic Metrics Section
    lines.append("## Semantic Metrics")
    lines.append("")
    lines.append("Semantic metrics measure meaning-level similarity using embeddings.")
    lines.append("")
    
    if semantic:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Mean Similarity | {semantic.get('semantic_similarity_mean', 0):.4f} |")
        lines.append(f"| Std | {semantic.get('semantic_similarity_std', 0):.4f} |")
        lines.append(f"| Min | {semantic.get('semantic_similarity_min', 0):.4f} |")
        lines.append(f"| Max | {semantic.get('semantic_similarity_max', 0):.4f} |")
        lines.append(f"| Median | {semantic.get('semantic_similarity_median', 0):.4f} |")
    else:
        lines.append("*No semantic metrics available*")
    
    lines.append("")
    
    # Retrieval Metrics Section
    lines.append("## Retrieval Metrics")
    lines.append("")
    lines.append("Retrieval metrics measure how well the system finds relevant documents.")
    lines.append("")
    
    if retrieval:
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        
        for key in retrieval.keys():
            if "_mean" in key:
                metric_name = key.replace("_mean", "").replace("@", "@").replace("_", " ").title()
                mean = retrieval.get(key, 0)
                std_key = key.replace("_mean", "_std")
                std = retrieval.get(std_key, 0)
                lines.append(f"| {metric_name} | {mean:.4f} | {std:.4f} |")
    else:
        lines.append("*No retrieval metrics available (requires relevant_product_ids in QA data)*")
    
    lines.append("")
    
    # LLM-as-Judge Section
    if llm_judge:
        lines.append("## LLM-as-Judge")
        lines.append("")
        lines.append(format_judge_results_for_report(llm_judge))
        lines.append("")
    
    # Ablation Study Section
    if results.get("ablation"):
        lines.append("## Ablation Study")
        lines.append("")
        
        ablation_data = results["ablation"]
        if isinstance(ablation_data, dict):
            # Convert to AblationResult objects for formatting
            ablation_results = {}
            for config_name, data in ablation_data.items():
                if isinstance(data, dict):
                    ablation_results[config_name] = AblationResult(
                        config_name=config_name,
                        lexical_metrics=data.get("lexical_metrics", {}),
                        semantic_metrics=data.get("semantic_metrics", {}),
                        retrieval_metrics=data.get("retrieval_metrics", {}),
                        avg_latency_ms=data.get("avg_latency_ms", 0),
                        total_time_s=data.get("total_time_s", 0),
                        num_samples=data.get("num_samples", 0)
                    )
            
            lines.append(format_ablation_comparison(ablation_results))
        
        lines.append("")
    
    # Agentic Analysis Section
    if results.get("agentic"):
        lines.append("## Agentic Analysis")
        lines.append("")
        
        agentic = results["agentic"]
        lines.append("### Tool Usage Statistics")
        lines.append("")
        
        if "tool_usage" in agentic:
            lines.append("| Tool | Call Count | Queries Using (%) |")
            lines.append("|------|------------|-------------------|")
            
            tool_usage = agentic["tool_usage"]
            tool_usage_pct = agentic.get("tool_usage_pct", {})
            
            for tool, count in tool_usage.items():
                pct = tool_usage_pct.get(tool, 0)
                lines.append(f"| {tool} | {count} | {pct:.1f}% |")
        
        lines.append("")
        
        if "avg_iterations" in agentic:
            lines.append(f"**Average Iterations per Query**: {agentic['avg_iterations']:.2f}")
        
        if "total_tool_calls" in agentic:
            lines.append(f"**Total Tool Calls**: {agentic['total_tool_calls']}")
        
        if "total_queries" in agentic:
            lines.append(f"**Total Queries Evaluated**: {agentic['total_queries']}")
        
        lines.append("")
    
    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    lines.append("This report summarizes the evaluation of the RAG pipeline. ")
    lines.append("Key findings should be interpreted in context of the specific use case and dataset.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by Tech Chatbot Evaluation Module*")
    
    return "\n".join(lines)


def save_markdown_report(
    results: Dict[str, Any],
    output_path: str = "eval_report.md",
    pipeline_name: str = "RAG Pipeline"
) -> str:
    """
    Generate and save markdown report.
    
    Args:
        results: Evaluation results dictionary
        output_path: Output file path
        pipeline_name: Name of the pipeline
        
    Returns:
        Path to saved file
    """
    markdown = generate_markdown_report(results, pipeline_name)
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"Markdown report saved to: {path}")
    return str(path)


def save_full_report(
    results: Dict[str, Any],
    output_dir: str = "eval_output",
    pipeline_name: str = "RAG Pipeline"
) -> Dict[str, str]:
    """
    Save both JSON and Markdown reports.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Output directory
        pipeline_name: Name of the pipeline
        
    Returns:
        Dictionary with paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = save_json_report(
        results,
        str(output_path / f"eval_results_{timestamp}.json")
    )
    
    md_path = save_markdown_report(
        results,
        str(output_path / f"eval_report_{timestamp}.md"),
        pipeline_name
    )
    
    return {
        "json": json_path,
        "markdown": md_path
    }


def print_summary(results: Dict[str, Any]):
    """
    Print a quick summary of evaluation results to console.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\nSamples evaluated: {results.get('num_samples', 'N/A')}")
    
    # Lexical
    lexical = results.get("lexical", {})
    if lexical:
        print("\nLexical Metrics:")
        print(f"  BLEU-1:    {lexical.get('bleu_1_mean', 0):.4f}")
        print(f"  ROUGE-L:   {lexical.get('rouge_l_f1_mean', 0):.4f}")
        print(f"  Token F1:  {lexical.get('token_f1_mean', 0):.4f}")
    
    # Semantic
    semantic = results.get("semantic", {})
    if semantic:
        print("\nSemantic Metrics:")
        print(f"  Similarity: {semantic.get('semantic_similarity_mean', 0):.4f}")
    
    # Retrieval
    retrieval = results.get("retrieval", {})
    if retrieval:
        print("\nRetrieval Metrics:")
        print(f"  Recall@10: {retrieval.get('recall@10_mean', 0):.4f}")
        print(f"  MRR:       {retrieval.get('mrr_mean', 0):.4f}")
    
    # LLM Judge
    llm_judge = results.get("llm_judge", {})
    if llm_judge and "faithfulness" in llm_judge:
        print("\nLLM-as-Judge:")
        print(f"  Faithfulness:  {llm_judge['faithfulness'].get('mean', 0):.2f}/5")
        print(f"  Relevance:     {llm_judge['relevance'].get('mean', 0):.2f}/5")
        print(f"  Completeness:  {llm_judge['completeness'].get('mean', 0):.2f}/5")
    
    print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    # Example results structure
    example_results = {
        "num_samples": 100,
        "lexical": {
            "bleu_1_mean": 0.32,
            "bleu_1_std": 0.15,
            "rouge_l_f1_mean": 0.41,
            "rouge_l_f1_std": 0.12,
            "token_f1_mean": 0.38,
            "token_f1_std": 0.14,
        },
        "semantic": {
            "semantic_similarity_mean": 0.78,
            "semantic_similarity_std": 0.10,
            "semantic_similarity_min": 0.45,
            "semantic_similarity_max": 0.95,
            "semantic_similarity_median": 0.80,
        },
        "retrieval": {
            "recall@10_mean": 0.72,
            "recall@10_std": 0.20,
            "mrr_mean": 0.65,
            "mrr_std": 0.25,
        },
        "llm_judge": {
            "faithfulness": {"mean": 4.2, "std": 0.8},
            "relevance": {"mean": 4.5, "std": 0.6},
            "completeness": {"mean": 3.8, "std": 0.9},
            "num_evaluated": 50,
            "num_errors": 2
        }
    }
    
    # Print summary
    print_summary(example_results)
    
    # Generate markdown
    markdown = generate_markdown_report(example_results, "Test Pipeline")
    print("\n--- Generated Markdown Preview ---\n")
    print(markdown[:1000] + "\n...")
