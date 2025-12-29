"""
Evaluation system with retrieval metrics and LLM-as-a-judge
"""
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from rag.vietnamese_retriever import VietnameseRetriever
from rag.reranker import QwenReranker
from rag.llm_service import GeminiLLMService, LLMJudgeService
from rag.unified_pipeline import UnifiedRAGPipeline
from rag.config import TOP_K_RETRIEVE, TOP_N_RERANK


class RetrievalMetrics:
    """
    Calculate retrieval metrics: MRR, NDCG@K, Recall@K, Precision@K
    """
    
    @staticmethod
    def mean_reciprocal_rank(relevant_docs: List[int], retrieved_docs: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            relevant_docs: List of relevant document IDs (ground truth)
            retrieved_docs: List of retrieved document IDs in order
            
        Returns:
            MRR score
        """
        if not relevant_docs:
            return 0.0
        
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K (NDCG@K)
        
        Args:
            relevant_docs: List of relevant document IDs (ground truth)
            retrieved_docs: List of retrieved document IDs in order
            k: Cutoff rank
            
        Returns:
            NDCG@K score
        """
        if not relevant_docs:
            return 0.0
        
        # Calculate DCG@K
        dcg = 0.0
        retrieved_k = retrieved_docs[:k]
        
        for rank, doc_id in enumerate(retrieved_k, 1):
            if doc_id in relevant_docs:
                # Relevance = 1 if relevant, 0 otherwise
                relevance = 1.0
                dcg += relevance / np.log2(rank + 1)
        
        # Calculate IDCG@K (ideal DCG)
        idcg = 0.0
        num_relevant = min(len(relevant_docs), k)
        for rank in range(1, num_relevant + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def recall_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
        """
        Calculate Recall@K
        
        Args:
            relevant_docs: List of relevant document IDs (ground truth)
            retrieved_docs: List of retrieved document IDs in order
            k: Cutoff rank
            
        Returns:
            Recall@K score
        """
        if not relevant_docs:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = retrieved_k.intersection(relevant_set)
        return len(intersection) / len(relevant_set)
    
    @staticmethod
    def precision_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            relevant_docs: List of relevant document IDs (ground truth)
            retrieved_docs: List of retrieved document IDs in order
            k: Cutoff rank
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = retrieved_k.intersection(relevant_set)
        return len(intersection) / k


class EvaluationSystem:
    """
    Comprehensive evaluation system
    """
    
    def __init__(self, retriever: Optional[VietnameseRetriever] = None,
                 reranker: Optional[QwenReranker] = None,
                 llm_service: Optional[GeminiLLMService] = None,
                 judge_service: Optional[LLMJudgeService] = None):
        """
        Initialize evaluation system
        
        Args:
            retriever: VietnameseRetriever instance
            reranker: QwenReranker instance
            llm_service: GeminiLLMService instance
            judge_service: LLMJudgeService instance
        """
        self.retriever = retriever or VietnameseRetriever()
        self.reranker = reranker or QwenReranker()
        self.llm_service = llm_service or GeminiLLMService()
        self.judge_service = judge_service or LLMJudgeService()
        self.retrieval_metrics = RetrievalMetrics()
    
    def evaluate_retrieval(self, query: str, relevant_doc_ids: List[int], 
                          k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate retrieval performance
        
        Args:
            query: Query text
            relevant_doc_ids: List of relevant document IDs (ground truth)
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary with retrieval metrics
        """
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(query, k=max(k_values))
        retrieved_doc_ids = [doc.metadata.get("doc_id") for doc in retrieved_docs]
        
        results = {}
        
        # Calculate metrics for each K
        for k in k_values:
            results[f"MRR@{k}"] = self.retrieval_metrics.mean_reciprocal_rank(
                relevant_doc_ids, retrieved_doc_ids
            )
            results[f"NDCG@{k}"] = self.retrieval_metrics.ndcg_at_k(
                relevant_doc_ids, retrieved_doc_ids, k
            )
            results[f"Recall@{k}"] = self.retrieval_metrics.recall_at_k(
                relevant_doc_ids, retrieved_doc_ids, k
            )
            results[f"Precision@{k}"] = self.retrieval_metrics.precision_at_k(
                relevant_doc_ids, retrieved_doc_ids, k
            )
        
        results["retrieved_count"] = len(retrieved_docs)
        results["relevant_count"] = len(relevant_doc_ids)
        
        return results
    
    def evaluate_end_to_end(self, query: str, ground_truth_answer: Optional[str] = None) -> Dict:
        """
        Evaluate end-to-end RAG performance using LLM-as-a-judge
        
        Args:
            query: Query text
            ground_truth_answer: Optional ground truth answer for comparison
            
        Returns:
            Dictionary with end-to-end evaluation scores
        """
        # Run RAG pipeline
        pipeline = UnifiedRAGPipeline(
            retriever=self.retriever,
            reranker=self.reranker,
            llm_service=self.llm_service
        )
        
        result = pipeline.query(query, use_history=False)
        answer = result["answer"]
        sources = result["sources"]
        
        # Format context from sources
        context = "\n".join([
            f"[Source {i+1}]\n{s['content_preview']}" 
            for i, s in enumerate(sources)
        ])
        
        # Evaluate with LLM-as-a-judge
        evaluation = self.judge_service.evaluate_comprehensive(query, answer, context)
        
        return {
            "query": query,
            "answer": answer,
            "sources_count": len(sources),
            "evaluation": evaluation,
            "ground_truth": ground_truth_answer
        }
    
    def evaluate_batch(self, eval_dataset: List[Dict], 
                      metrics: List[str] = ["retrieval", "end_to_end"]) -> Dict:
        """
        Evaluate on a batch of queries
        
        Args:
            eval_dataset: List of evaluation examples, each with:
                - query: Query text
                - relevant_doc_ids: List of relevant document IDs (for retrieval)
                - ground_truth_answer: Optional ground truth answer (for end-to-end)
            metrics: List of metrics to calculate
        
        Returns:
            Dictionary with aggregated results
        """
        results = {
            "retrieval": [],
            "end_to_end": []
        }
        
        for example in eval_dataset:
            query = example.get("query", "")
            relevant_doc_ids = example.get("relevant_doc_ids", [])
            ground_truth_answer = example.get("ground_truth_answer")
            
            if "retrieval" in metrics and relevant_doc_ids:
                retrieval_result = self.evaluate_retrieval(query, relevant_doc_ids)
                retrieval_result["query"] = query
                results["retrieval"].append(retrieval_result)
            
            if "end_to_end" in metrics:
                e2e_result = self.evaluate_end_to_end(query, ground_truth_answer)
                results["end_to_end"].append(e2e_result)
        
        # Aggregate results
        aggregated = {}
        
        if results["retrieval"]:
            aggregated["retrieval"] = self._aggregate_retrieval_metrics(results["retrieval"])
        
        if results["end_to_end"]:
            aggregated["end_to_end"] = self._aggregate_e2e_metrics(results["end_to_end"])
        
        aggregated["total_examples"] = len(eval_dataset)
        
        return aggregated
    
    def _aggregate_retrieval_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate retrieval metrics across examples"""
        aggregated = {}
        
        # Get all metric keys
        metric_keys = [k for k in results[0].keys() if k not in ["query", "retrieved_count", "relevant_count"]]
        
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
        
        return aggregated
    
    def _aggregate_e2e_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate end-to-end metrics across examples"""
        relevance_scores = []
        faithfulness_scores = []
        utilization_scores = []
        average_scores = []
        
        for result in results:
            eval_data = result.get("evaluation", {})
            if "relevance" in eval_data:
                relevance_scores.append(eval_data["relevance"]["score"])
            if "faithfulness" in eval_data:
                faithfulness_scores.append(eval_data["faithfulness"]["score"])
            if "context_utilization" in eval_data:
                utilization_scores.append(eval_data["context_utilization"]["score"])
            if "average_score" in eval_data:
                average_scores.append(eval_data["average_score"])
        
        aggregated = {}
        if relevance_scores:
            aggregated["relevance_mean"] = np.mean(relevance_scores)
            aggregated["relevance_std"] = np.std(relevance_scores)
        if faithfulness_scores:
            aggregated["faithfulness_mean"] = np.mean(faithfulness_scores)
            aggregated["faithfulness_std"] = np.std(faithfulness_scores)
        if utilization_scores:
            aggregated["utilization_mean"] = np.mean(utilization_scores)
            aggregated["utilization_std"] = np.std(utilization_scores)
        if average_scores:
            aggregated["average_score_mean"] = np.mean(average_scores)
            aggregated["average_score_std"] = np.std(average_scores)
        
        return aggregated
    
    def load_eval_dataset(self, file_path: str) -> List[Dict]:
        """
        Load evaluation dataset from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of evaluation examples
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "examples" in data:
            return data["examples"]
        else:
            raise ValueError("Invalid dataset format")


def run_evaluation(eval_dataset_path: str, output_path: Optional[str] = None):
    """
    Run evaluation and save results
    
    Args:
        eval_dataset_path: Path to evaluation dataset JSON
        output_path: Optional path to save results
    """
    evaluator = EvaluationSystem()
    
    print(f"Loading evaluation dataset from {eval_dataset_path}")
    eval_dataset = evaluator.load_eval_dataset(eval_dataset_path)
    
    print(f"Running evaluation on {len(eval_dataset)} examples...")
    results = evaluator.evaluate_batch(eval_dataset, metrics=["retrieval", "end_to_end"])
    
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset JSON")
    parser.add_argument("--output", help="Path to save results JSON")
    
    args = parser.parse_args()
    
    run_evaluation(args.dataset, args.output)