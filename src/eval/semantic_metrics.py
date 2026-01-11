"""
Semantic similarity evaluation - Uses local embedding model (FREE).

Reuses the Vietnamese embedding model from chunk_embedding.embedding.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple

# Import the existing Embedder class
from src.chunk_embedding.embedding import Embedder


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (0 to 1 for normalized vectors)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(vec1 - vec2))


class SemanticEvaluator:
    """
    Evaluate semantic similarity using embeddings.
    
    Uses the same Vietnamese embedding model as the RAG system
    to ensure consistency.
    """
    
    def __init__(
        self,
        model_name: str = "dangvantuan/vietnamese-document-embedding",
        device: Optional[str] = None
    ):
        """
        Initialize semantic evaluator.
        
        Args:
            model_name: Embedding model name
            device: Device to use (auto-detected if None)
        """
        print(f"Initializing SemanticEvaluator with {model_name}")
        self.embedder = Embedder(model_name=model_name, device=device)
        print("SemanticEvaluator ready")
    
    def evaluate_single(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute semantic similarity for one pair.
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            Dictionary with similarity metrics
        """
        if not prediction or not reference:
            return {
                "cosine_similarity": 0.0,
                "euclidean_distance": float('inf')
            }
        
        # Embed both texts
        embeddings = self.embedder.embed([prediction, reference])
        pred_emb = np.array(embeddings[0])
        ref_emb = np.array(embeddings[1])
        
        return {
            "cosine_similarity": cosine_similarity(pred_emb, ref_emb),
            "euclidean_distance": euclidean_distance(pred_emb, ref_emb)
        }
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate a batch and return statistics.
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            batch_size: Batch size for embedding
            
        Returns:
            Dictionary with aggregate statistics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        if not predictions:
            return {}
        
        print(f"Computing semantic similarity for {len(predictions)} samples...")
        
        # Batch encode all texts for efficiency
        all_texts = predictions + references
        
        # Encode in batches to avoid OOM
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            batch_embeddings = self.embedder.embed(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Split back into predictions and references
        pred_embeddings = np.array(all_embeddings[:len(predictions)])
        ref_embeddings = np.array(all_embeddings[len(predictions):])
        
        # Compute pairwise similarities
        similarities = []
        distances = []
        
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            sim = cosine_similarity(pred_emb, ref_emb)
            dist = euclidean_distance(pred_emb, ref_emb)
            similarities.append(sim)
            distances.append(dist)
        
        similarities = np.array(similarities)
        distances = np.array(distances)
        
        return {
            "semantic_similarity_mean": float(np.mean(similarities)),
            "semantic_similarity_std": float(np.std(similarities)),
            "semantic_similarity_min": float(np.min(similarities)),
            "semantic_similarity_max": float(np.max(similarities)),
            "semantic_similarity_median": float(np.median(similarities)),
            "euclidean_distance_mean": float(np.mean(distances)),
            "euclidean_distance_std": float(np.std(distances)),
            "num_samples": len(predictions)
        }
    
    def evaluate_with_details(
        self,
        predictions: List[str],
        references: List[str],
        batch_size: int = 32
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate batch and return both aggregate and per-sample results.
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            batch_size: Batch size for embedding
            
        Returns:
            Tuple of (aggregate_metrics, per_sample_metrics)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        if not predictions:
            return {}, []
        
        # Batch encode all texts
        all_texts = predictions + references
        all_embeddings = []
        
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            batch_embeddings = self.embedder.embed(batch)
            all_embeddings.extend(batch_embeddings)
        
        pred_embeddings = np.array(all_embeddings[:len(predictions)])
        ref_embeddings = np.array(all_embeddings[len(predictions):])
        
        # Per-sample metrics
        per_sample = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            per_sample.append({
                "cosine_similarity": cosine_similarity(pred_emb, ref_emb),
                "euclidean_distance": euclidean_distance(pred_emb, ref_emb)
            })
        
        # Aggregate metrics
        similarities = [p["cosine_similarity"] for p in per_sample]
        distances = [p["euclidean_distance"] for p in per_sample]
        
        aggregate = {
            "semantic_similarity_mean": float(np.mean(similarities)),
            "semantic_similarity_std": float(np.std(similarities)),
            "semantic_similarity_min": float(np.min(similarities)),
            "semantic_similarity_max": float(np.max(similarities)),
            "semantic_similarity_median": float(np.median(similarities)),
            "euclidean_distance_mean": float(np.mean(distances)),
            "euclidean_distance_std": float(np.std(distances)),
            "num_samples": len(predictions)
        }
        
        return aggregate, per_sample
    
    def compute_answer_context_similarity(
        self,
        answers: List[str],
        contexts: List[str]
    ) -> Dict[str, float]:
        """
        Compute similarity between answers and their retrieved contexts.
        
        This can help identify if answers are grounded in context.
        
        Args:
            answers: Generated answers
            contexts: Retrieved contexts
            
        Returns:
            Similarity statistics
        """
        if len(answers) != len(contexts):
            raise ValueError("Answers and contexts must have same length")
        
        results = self.evaluate_batch(answers, contexts)
        
        # Rename keys for clarity
        return {
            "answer_context_similarity_mean": results["semantic_similarity_mean"],
            "answer_context_similarity_std": results["semantic_similarity_std"],
            "answer_context_similarity_min": results["semantic_similarity_min"],
            "answer_context_similarity_max": results["semantic_similarity_max"],
        }


# Example usage and testing
if __name__ == "__main__":
    evaluator = SemanticEvaluator()
    
    # Test cases
    test_cases = [
        {
            "prediction": "iPhone 15 Pro Max có giá 28.990.000 VNĐ, được trang bị chip A17 Pro",
            "reference": "Giá iPhone 15 Pro Max là 28,990,000 đồng với chip Apple A17 Pro"
        },
        {
            "prediction": "Samsung Galaxy S24 Ultra sử dụng chip Snapdragon 8 Gen 3",
            "reference": "Galaxy S24 Ultra dùng chip Snapdragon 8 Gen 3 for Galaxy"
        },
        {
            "prediction": "Điện thoại này hoàn toàn không liên quan gì",
            "reference": "Sản phẩm được trang bị màn hình AMOLED kích thước 6.7 inch"
        }
    ]
    
    print("=" * 60)
    print("SEMANTIC SIMILARITY EVALUATION TEST")
    print("=" * 60)
    
    predictions = [tc["prediction"] for tc in test_cases]
    references = [tc["reference"] for tc in test_cases]
    
    # Per-sample evaluation
    for i, tc in enumerate(test_cases):
        print(f"\nSample {i+1}:")
        print(f"  Prediction: {tc['prediction'][:50]}...")
        print(f"  Reference:  {tc['reference'][:50]}...")
        
        result = evaluator.evaluate_single(tc["prediction"], tc["reference"])
        print(f"  Cosine Similarity: {result['cosine_similarity']:.4f}")
        print(f"  Euclidean Distance: {result['euclidean_distance']:.4f}")
    
    # Batch evaluation
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    
    batch_results = evaluator.evaluate_batch(predictions, references)
    for metric, value in batch_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
