#!/usr/bin/env python3
"""
Main evaluation runner - CLI interface for running RAG evaluations.

Usage (from project root):
    # FREE evaluation only (no LLM judge)
    python -m src.eval.run_evaluation --qa-file data/qa_pairs.json --no-llm

    # Full evaluation with LLM judge on 50 samples
    python -m src.eval.run_evaluation --qa-file data/qa_pairs.json --llm-samples 50

    # Compare UnifiedRAG and AgenticRAG
    python -m src.eval.run_evaluation --qa-file data/qa_pairs.json --compare-agentic

    # Run ablation study
    python -m src.eval.run_evaluation --qa-file data/qa_pairs.json --ablation
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from .data_loader import load_qa_data, validate_qa_data, split_by_type, QASample, EvalPrediction
from .lexical_metrics import LexicalEvaluator
from .semantic_metrics import SemanticEvaluator
from .retrieval_metrics import RetrievalEvaluator
from .llm_judge import LLMJudge
from .ablation import AblationRunner, AblationConfig
from .report_generator import save_full_report, print_summary


def run_pipeline_evaluation(
    pipeline,
    samples: List[QASample],
    pipeline_name: str = "Pipeline",
    verbose: bool = True
) -> tuple[List[EvalPrediction], float]:
    """
    Run evaluation on a pipeline and collect predictions.
    
    Args:
        pipeline: RAG pipeline with query_for_eval method
        samples: QA samples to evaluate
        pipeline_name: Name for logging
        verbose: Print progress
        
    Returns:
        Tuple of (predictions, total_time)
    """
    predictions = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {pipeline_name} on {len(samples)} samples")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        if verbose:
            print(f"  Processing {i+1}/{len(samples)}: {sample.question[:50]}...", end="\r")
        
        try:
            result = pipeline.query_for_eval(sample.question)
            
            pred = EvalPrediction(
                query=sample.question,
                ground_truth=sample.answer,
                prediction=result.get("answer", ""),
                context=result.get("context", ""),
                retrieved_ids=result.get("retrieved_ids", []),
                reranked_ids=result.get("reranked_ids", []),
                relevant_ids=sample.relevant_product_ids,
                sources=result.get("sources", []),
                tool_log=result.get("tool_log", []),
                iterations=result.get("iterations", 0)
            )
            predictions.append(pred)
            
        except Exception as e:
            print(f"\n  Error on sample {i+1}: {e}")
            # Add empty prediction
            pred = EvalPrediction(
                query=sample.question,
                ground_truth=sample.answer,
                prediction="[ERROR]",
                relevant_ids=sample.relevant_product_ids
            )
            predictions.append(pred)
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n  Completed in {total_time:.1f}s ({total_time/len(samples)*1000:.0f}ms/query)")
    
    return predictions, total_time


def evaluate_predictions(
    predictions: List[EvalPrediction],
    run_semantic: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run all evaluators on predictions.
    
    Args:
        predictions: List of predictions to evaluate
        run_semantic: Run semantic evaluation (slower but FREE)
        verbose: Print progress
        
    Returns:
        Dictionary of evaluation results
    """
    results = {}
    
    pred_texts = [p.prediction for p in predictions]
    ref_texts = [p.ground_truth for p in predictions]
    
    # Lexical Evaluation (FREE)
    if verbose:
        print("\nRunning Lexical Evaluation...")
    lexical_eval = LexicalEvaluator()
    results["lexical"] = lexical_eval.evaluate_batch(pred_texts, ref_texts)
    
    # Semantic Evaluation (FREE but slower)
    if run_semantic:
        if verbose:
            print("Running Semantic Evaluation...")
        semantic_eval = SemanticEvaluator()
        results["semantic"] = semantic_eval.evaluate_batch(pred_texts, ref_texts)
    
    # Retrieval Evaluation (FREE, requires relevant_ids)
    has_relevant = any(p.relevant_ids for p in predictions)
    if has_relevant:
        if verbose:
            print("Running Retrieval Evaluation...")
        retrieval_eval = RetrievalEvaluator()
        all_retrieved = [p.retrieved_ids for p in predictions]
        all_relevant = [p.relevant_ids for p in predictions]
        results["retrieval"] = retrieval_eval.evaluate_batch(all_retrieved, all_relevant)
    
    results["num_samples"] = len(predictions)
    
    return results


def run_llm_judge_evaluation(
    predictions: List[EvalPrediction],
    sample_size: int = 50,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run LLM-as-Judge evaluation on a sample of predictions.
    
    Args:
        predictions: List of predictions to evaluate
        sample_size: Number of samples to evaluate (budget optimization)
        verbose: Print progress
        
    Returns:
        LLM judge results
    """
    if verbose:
        print(f"\nRunning LLM-as-Judge on {sample_size} samples...")
    
    # Convert predictions to format expected by LLMJudge
    samples = [
        {
            "query": p.query,
            "answer": p.prediction,
            "context": p.context,
            "ground_truth": p.ground_truth
        }
        for p in predictions
    ]
    
    judge = LLMJudge()
    return judge.evaluate_batch(samples, sample_size=sample_size)


def compute_agentic_metrics(predictions: List[EvalPrediction]) -> Dict[str, Any]:
    """
    Compute agentic-specific metrics from predictions.
    
    Args:
        predictions: List of predictions with tool_log and iterations
        
    Returns:
        Agentic metrics including tool usage counts and percentages
    """
    if not predictions:
        return {
            "tool_usage": {},
            "tool_usage_pct": {},
            "avg_iterations": 0,
            "total_tool_calls": 0,
            "total_queries": 0
        }
    
    # Count tool usage
    tool_usage = {}
    total_iterations = 0
    
    for pred in predictions:
        total_iterations += pred.iterations
        for tool in pred.tool_log:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
    
    avg_iterations = total_iterations / len(predictions)
    total_tool_calls = sum(tool_usage.values())
    
    # Calculate tool usage percentage (% of queries that used each tool)
    tool_usage_pct = {}
    for tool, count in tool_usage.items():
        # Percentage of queries that used this tool at least once
        queries_with_tool = sum(1 for p in predictions if tool in p.tool_log)
        tool_usage_pct[tool] = (queries_with_tool / len(predictions)) * 100
    
    return {
        "tool_usage": tool_usage,
        "tool_usage_pct": tool_usage_pct,
        "avg_iterations": avg_iterations,
        "total_tool_calls": total_tool_calls,
        "total_queries": len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from project root):
  # FREE evaluation only (no API cost)
  python -m src.eval.run_evaluation --qa-file data/qa_pairs.json --no-llm

  # Full evaluation with LLM judge
  python -m src.eval.run_evaluation --qa-file data/qa_pairs.json --llm-samples 50

  # Compare with AgenticRAG
  python -m src.eval.run_evaluation --qa-file data/qa_pairs.json --compare-agentic

  # Run ablation study
  python -m src.eval.run_evaluation --qa-file data/qa_pairs.json --ablation
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--qa-file",
        type=str,
        required=True,
        help="Path to QA pairs JSON file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_output",
        help="Output directory for reports (default: eval_output)"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM-as-Judge evaluation (saves API cost)"
    )
    
    parser.add_argument(
        "--llm-samples",
        type=int,
        default=50,
        help="Number of samples for LLM judge (default: 50)"
    )
    
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic similarity evaluation (faster)"
    )
    
    parser.add_argument(
        "--compare-agentic",
        action="store_true",
        help="Also evaluate AgenticRAG pipeline"
    )
    
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study (baseline vs no-decomposition)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of QA samples to evaluate (for testing)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Load QA data
    if verbose:
        print(f"\nLoading QA data from: {args.qa_file}")
    
    try:
        samples = load_qa_data(args.qa_file)
    except FileNotFoundError:
        print(f"Error: QA file not found: {args.qa_file}")
        sys.exit(1)
    
    # Validate data
    if verbose:
        validate_qa_data(samples)
        split_by_type(samples)
    
    # Limit samples if requested
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[:args.max_samples]
        if verbose:
            print(f"\nLimited to {args.max_samples} samples for evaluation")
    
    # Initialize pipelines
    if verbose:
        print("\nInitializing RAG pipelines...")
    
    from src.rag.unified_pipeline import UnifiedRAGPipeline
    unified_pipeline = UnifiedRAGPipeline()
    
    agentic_pipeline = None
    if args.compare_agentic:
        from src.rag.agentic_pipeline import AgenticRAGPipeline
        agentic_pipeline = AgenticRAGPipeline()
    
    # Results container
    all_results: Dict[str, Any] = {
        "num_samples": len(samples),
        "config": {
            "qa_file": args.qa_file,
            "llm_judge": not args.no_llm,
            "llm_samples": args.llm_samples if not args.no_llm else 0,
            "semantic": not args.no_semantic,
            "compare_agentic": args.compare_agentic,
            "ablation": args.ablation
        }
    }
    
    # Run ablation study if requested
    if args.ablation:
        if verbose:
            print("\n" + "="*60)
            print("RUNNING ABLATION STUDY")
            print("="*60)
        
        ablation_runner = AblationRunner(
            unified_pipeline=unified_pipeline,
            agentic_pipeline=agentic_pipeline,
            run_semantic=not args.no_semantic
        )
        
        configs = [AblationConfig.BASELINE, AblationConfig.NO_DECOMPOSITION]
        if agentic_pipeline:
            configs.append(AblationConfig.AGENTIC)
        
        ablation_results = ablation_runner.run_ablation_study(
            samples=samples,
            configs=configs,
            verbose=verbose
        )
        
        # Convert to serializable format
        all_results["ablation"] = {
            name: result.to_dict() 
            for name, result in ablation_results.items()
        }
        
        # Use baseline results as main results
        if "baseline" in ablation_results:
            baseline = ablation_results["baseline"]
            all_results["lexical"] = baseline.lexical_metrics
            all_results["semantic"] = baseline.semantic_metrics
            all_results["retrieval"] = baseline.retrieval_metrics
    
    else:
        # Standard evaluation (non-ablation)
        predictions, eval_time = run_pipeline_evaluation(
            unified_pipeline,
            samples,
            pipeline_name="UnifiedRAG",
            verbose=verbose
        )
        
        # Run evaluations
        eval_results = evaluate_predictions(
            predictions,
            run_semantic=not args.no_semantic,
            verbose=verbose
        )
        
        all_results.update(eval_results)
        all_results["eval_time_s"] = eval_time
        
        # AgenticRAG comparison
        if args.compare_agentic and agentic_pipeline:
            agentic_preds, agentic_time = run_pipeline_evaluation(
                agentic_pipeline,
                samples,
                pipeline_name="AgenticRAG",
                verbose=verbose
            )
            
            agentic_results = evaluate_predictions(
                agentic_preds,
                run_semantic=not args.no_semantic,
                verbose=verbose
            )
            
            # Compute agentic-specific metrics
            agentic_metrics = compute_agentic_metrics(agentic_preds)
            
            all_results["agentic"] = {
                **agentic_metrics,
                "lexical": agentic_results.get("lexical", {}),
                "semantic": agentic_results.get("semantic", {}),
                "retrieval": agentic_results.get("retrieval", {}),
                "eval_time_s": agentic_time
            }
    
    # LLM-as-Judge (if enabled)
    if not args.no_llm:
        # Need predictions for LLM judge
        if not args.ablation:
            llm_results = run_llm_judge_evaluation(
                predictions,
                sample_size=args.llm_samples,
                verbose=verbose
            )
            all_results["llm_judge"] = llm_results
        else:
            if verbose:
                print("\nNote: LLM judge skipped for ablation study to save cost")
    
    # Print summary
    if verbose:
        print_summary(all_results)
    
    # Save reports
    if verbose:
        print(f"\nSaving reports to: {args.output_dir}")
    
    saved_files = save_full_report(
        all_results,
        output_dir=args.output_dir,
        pipeline_name="UnifiedRAG + AgenticRAG" if args.compare_agentic else "UnifiedRAG"
    )
    
    if verbose:
        print("\nEvaluation complete!")
        print(f"  JSON report: {saved_files['json']}")
        print(f"  Markdown report: {saved_files['markdown']}")
    
    return all_results


if __name__ == "__main__":
    main()
