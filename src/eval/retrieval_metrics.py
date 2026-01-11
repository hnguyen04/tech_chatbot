"""
Retrieval evaluation metrics - 100% FREE.

Metrics:
- Recall@K: % of relevant products in top-K retrieved
- Precision@K: % of retrieved products that are relevant
- Hit Rate@K: Binary - did ANY relevant product appear?
- MRR: Mean Reciprocal Rank of first relevant result
- nDCG@K: Normalized Discounted Cumulative Gain
"""
import math
from typing import List, Dict, Set, Optional, Tuple


def compute_recall_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int = 10
) -> float:
    """
    Compute Recall@K.
    
    Recall = |Retrieved ∩ Relevant| / |Relevant|
    
    Args:
        retrieved_ids: List of retrieved product IDs (ordered by rank)
        relevant_ids: List of relevant product IDs (ground truth)
        k: Consider only top-K retrieved
        
    Returns:
        Recall@K score (0 to 1)
    """
    if not relevant_ids:
        return 0.0  # No relevant items to find
    
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = retrieved_set & relevant_set
    
    return len(hits) / len(relevant_set)


def compute_precision_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int = 10
) -> float:
    """
    Compute Precision@K.
    
    Precision = |Retrieved ∩ Relevant| / |Retrieved@K|
    
    Args:
        retrieved_ids: List of retrieved product IDs (ordered by rank)
        relevant_ids: List of relevant product IDs (ground truth)
        k: Consider only top-K retrieved
        
    Returns:
        Precision@K score (0 to 1)
    """
    if not retrieved_ids:
        return 0.0
    
    # Only consider top-K
    top_k_retrieved = retrieved_ids[:k]
    
    if not top_k_retrieved:
        return 0.0
    
    retrieved_set = set(top_k_retrieved)
    relevant_set = set(relevant_ids)
    
    hits = retrieved_set & relevant_set
    
    return len(hits) / len(top_k_retrieved)


def compute_hit_rate_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int = 10
) -> float:
    """
    Compute Hit Rate@K (binary).
    
    1 if any relevant item is in top-K, 0 otherwise.
    
    Args:
        retrieved_ids: List of retrieved product IDs (ordered by rank)
        relevant_ids: List of relevant product IDs (ground truth)
        k: Consider only top-K retrieved
        
    Returns:
        Hit Rate@K (0 or 1)
    """
    if not relevant_ids:
        return 0.0
    
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    hits = retrieved_set & relevant_set
    
    return 1.0 if hits else 0.0


def compute_mrr(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: Optional[int] = None
) -> float:
    """
    Compute Mean Reciprocal Rank (for single query).
    
    MRR = 1 / rank_of_first_relevant_item
    
    Args:
        retrieved_ids: List of retrieved product IDs (ordered by rank)
        relevant_ids: List of relevant product IDs (ground truth)
        k: Consider only top-K (None for all)
        
    Returns:
        Reciprocal Rank (0 to 1)
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    
    relevant_set = set(relevant_ids)
    search_list = retrieved_ids[:k] if k else retrieved_ids
    
    for i, rid in enumerate(search_list, 1):
        if rid in relevant_set:
            return 1.0 / i
    
    return 0.0


def compute_dcg_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int = 10
) -> float:
    """
    Compute Discounted Cumulative Gain@K.
    
    DCG = Σ rel_i / log2(i + 1)
    
    Uses binary relevance (1 if relevant, 0 otherwise).
    
    Args:
        retrieved_ids: List of retrieved product IDs (ordered by rank)
        relevant_ids: List of relevant product IDs (ground truth)
        k: Consider only top-K retrieved
        
    Returns:
        DCG@K score
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    
    relevant_set = set(relevant_ids)
    dcg = 0.0
    
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_set:
            # Binary relevance: rel_i = 1 if relevant, 0 otherwise
            dcg += 1.0 / math.log2(i + 1)
    
    return dcg


def compute_ndcg_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int = 10
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain@K.
    
    nDCG = DCG@K / IDCG@K
    
    Args:
        retrieved_ids: List of retrieved product IDs (ordered by rank)
        relevant_ids: List of relevant product IDs (ground truth)
        k: Consider only top-K retrieved
        
    Returns:
        nDCG@K score (0 to 1)
    """
    if not relevant_ids:
        return 0.0
    
    # Actual DCG
    dcg = compute_dcg_at_k(retrieved_ids, relevant_ids, k)
    
    # Ideal DCG (all relevant items ranked first)
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


class RetrievalEvaluator:
    """
    Evaluate retrieval quality using standard IR metrics.
    
    All computations are local and FREE.
    """
    
    def __init__(self, default_k: int = 10):
        """
        Initialize evaluator.
        
        Args:
            default_k: Default K value for @K metrics
        """
        self.default_k = default_k
    
    def evaluate_single(
        self,
        retrieved_ids: List[int],
        relevant_ids: List[int],
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval for one query.
        
        Args:
            retrieved_ids: List of retrieved product IDs (ordered by rank)
            relevant_ids: List of relevant product IDs (ground truth)
            k: K value for @K metrics (uses default if None)
            
        Returns:
            Dictionary of metrics
        """
        k = k or self.default_k
        
        return {
            f"recall@{k}": compute_recall_at_k(retrieved_ids, relevant_ids, k),
            f"precision@{k}": compute_precision_at_k(retrieved_ids, relevant_ids, k),
            f"hit_rate@{k}": compute_hit_rate_at_k(retrieved_ids, relevant_ids, k),
            "mrr": compute_mrr(retrieved_ids, relevant_ids, k),
            f"ndcg@{k}": compute_ndcg_at_k(retrieved_ids, relevant_ids, k),
        }
    
    def evaluate_batch(
        self,
        all_retrieved: List[List[int]],
        all_relevant: List[List[int]],
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate batch of queries.
        
        Args:
            all_retrieved: List of retrieved ID lists (one per query)
            all_relevant: List of relevant ID lists (one per query)
            k: K value for @K metrics
            
        Returns:
            Dictionary of averaged metrics
        """
        if len(all_retrieved) != len(all_relevant):
            raise ValueError("Retrieved and relevant lists must have same length")
        
        if not all_retrieved:
            return {}
        
        k = k or self.default_k
        
        # Compute per-query metrics
        all_results = []
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            result = self.evaluate_single(retrieved, relevant, k)
            all_results.append(result)
        
        # Average across all queries
        avg_results = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            avg_results[f"{key}_mean"] = sum(values) / len(values)
            
            # Compute std for key metrics
            mean = avg_results[f"{key}_mean"]
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            avg_results[f"{key}_std"] = variance ** 0.5
        
        avg_results["num_queries"] = len(all_retrieved)
        
        return avg_results
    
    def evaluate_with_details(
        self,
        all_retrieved: List[List[int]],
        all_relevant: List[List[int]],
        k: Optional[int] = None
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate batch and return both aggregate and per-query results.
        
        Args:
            all_retrieved: List of retrieved ID lists
            all_relevant: List of relevant ID lists
            k: K value for @K metrics
            
        Returns:
            Tuple of (aggregate_metrics, per_query_metrics)
        """
        k = k or self.default_k
        
        per_query = []
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            result = self.evaluate_single(retrieved, relevant, k)
            per_query.append(result)
        
        aggregate = self.evaluate_batch(all_retrieved, all_relevant, k)
        
        return aggregate, per_query
    
    def evaluate_at_multiple_k(
        self,
        all_retrieved: List[List[int]],
        all_relevant: List[List[int]],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate at multiple K values.
        
        Args:
            all_retrieved: List of retrieved ID lists
            all_relevant: List of relevant ID lists
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary mapping K to metrics
        """
        results = {}
        for k in k_values:
            results[f"k={k}"] = self.evaluate_batch(all_retrieved, all_relevant, k)
        return results


# Example usage and testing
if __name__ == "__main__":
    evaluator = RetrievalEvaluator(default_k=10)
    
    # Test cases
    test_cases = [
        {
            "retrieved": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Retrieved IDs
            "relevant": [2, 5, 8],  # Relevant IDs
            "description": "3 relevant items in top-10"
        },
        {
            "retrieved": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "relevant": [11, 12, 13],  # None in top-10
            "description": "No relevant items in top-10"
        },
        {
            "retrieved": [1, 2, 3, 4, 5],
            "relevant": [1, 2],  # First 2 are relevant
            "description": "Relevant items at top positions"
        }
    ]
    
    print("=" * 60)
    print("RETRIEVAL METRICS EVALUATION TEST")
    print("=" * 60)
    
    for i, tc in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {tc['description']}")
        print(f"  Retrieved: {tc['retrieved'][:5]}...")
        print(f"  Relevant:  {tc['relevant']}")
        
        result = evaluator.evaluate_single(tc["retrieved"], tc["relevant"])
        for metric, value in result.items():
            print(f"  {metric}: {value:.4f}")
    
    # Batch evaluation
    print("\n" + "=" * 60)
    print("BATCH EVALUATION")
    print("=" * 60)
    
    all_retrieved = [tc["retrieved"] for tc in test_cases]
    all_relevant = [tc["relevant"] for tc in test_cases]
    
    batch_results = evaluator.evaluate_batch(all_retrieved, all_relevant)
    for metric, value in batch_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Multiple K evaluation
    print("\n" + "=" * 60)
    print("EVALUATION AT MULTIPLE K VALUES")
    print("=" * 60)
    
    multi_k_results = evaluator.evaluate_at_multiple_k(
        all_retrieved, all_relevant, k_values=[3, 5, 10]
    )
    
    for k_label, metrics in multi_k_results.items():
        print(f"\n{k_label}:")
        for metric, value in metrics.items():
            if isinstance(value, float) and "mean" in metric:
                print(f"  {metric}: {value:.4f}")
