"""
Lexical evaluation metrics - 100% FREE, runs locally on CPU.

Metrics:
- BLEU-1: Unigram precision with brevity penalty
- ROUGE-L: Longest common subsequence F1
- Token F1: Word-level precision/recall/F1
"""
import re
from collections import Counter
from typing import List, Dict, Tuple


def tokenize_vietnamese(text: str) -> List[str]:
    """
    Simple Vietnamese tokenizer (word-level).
    
    Handles Vietnamese Unicode and common punctuation.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens (lowercase)
    """
    if not text:
        return []
    
    # Lowercase
    text = text.lower()
    
    # Split by whitespace and punctuation, keeping Vietnamese characters
    # This regex matches word characters including Vietnamese diacritics
    tokens = re.findall(r'\b[\w\u00C0-\u024F\u1E00-\u1EFF]+\b', text, re.UNICODE)
    
    return tokens


def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Compute n-gram counts.
    
    Args:
        tokens: List of tokens
        n: N-gram size
        
    Returns:
        Counter of n-grams
    """
    if len(tokens) < n:
        return Counter()
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    return Counter(ngrams)


def compute_bleu_n(
    pred_tokens: List[str], 
    ref_tokens: List[str],
    n: int = 1
) -> float:
    """
    Compute BLEU-n score (modified precision).
    
    Args:
        pred_tokens: Prediction tokens
        ref_tokens: Reference tokens
        n: N-gram size (1 for BLEU-1)
        
    Returns:
        BLEU-n precision score
    """
    if not pred_tokens or len(pred_tokens) < n:
        return 0.0
    
    pred_ngrams = compute_ngrams(pred_tokens, n)
    ref_ngrams = compute_ngrams(ref_tokens, n)
    
    # Clipped counts (min of pred count and ref count for each n-gram)
    clipped_count = 0
    total_count = sum(pred_ngrams.values())
    
    for ngram, count in pred_ngrams.items():
        clipped_count += min(count, ref_ngrams.get(ngram, 0))
    
    if total_count == 0:
        return 0.0
    
    precision = clipped_count / total_count
    
    return precision


def compute_brevity_penalty(pred_len: int, ref_len: int) -> float:
    """
    Compute brevity penalty for BLEU.
    
    Args:
        pred_len: Prediction length
        ref_len: Reference length
        
    Returns:
        Brevity penalty (0 to 1)
    """
    if pred_len == 0:
        return 0.0
    
    if pred_len >= ref_len:
        return 1.0
    
    return pred_len / ref_len  # Simplified BP


def compute_bleu_1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """
    Compute BLEU-1 score with brevity penalty.
    
    Args:
        pred_tokens: Prediction tokens
        ref_tokens: Reference tokens
        
    Returns:
        BLEU-1 score (0 to 1)
    """
    precision = compute_bleu_n(pred_tokens, ref_tokens, n=1)
    bp = compute_brevity_penalty(len(pred_tokens), len(ref_tokens))
    
    return bp * precision


def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute Longest Common Subsequence length using dynamic programming.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        LCS length
    """
    if not seq1 or not seq2:
        return 0
    
    m, n = len(seq1), len(seq2)
    
    # Optimize space: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev
    
    return prev[n]


def compute_rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L score (Longest Common Subsequence based).
    
    Args:
        pred_tokens: Prediction tokens
        ref_tokens: Reference tokens
        
    Returns:
        Dictionary with precision, recall, and F1
    """
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    lcs_length = compute_lcs_length(pred_tokens, ref_tokens)
    
    precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_token_f1(pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
    """
    Compute token-level precision, recall, and F1.
    
    Args:
        pred_tokens: Prediction tokens
        ref_tokens: Reference tokens
        
    Returns:
        Dictionary with precision, recall, and F1
    """
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    # Common tokens (intersection with min counts)
    common = sum((pred_counter & ref_counter).values())
    
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}


class LexicalEvaluator:
    """
    Evaluate generated answers using lexical metrics.
    
    All computations are local and FREE.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_single(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Evaluate a single prediction-reference pair.
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            Dictionary of metrics
        """
        pred_tokens = tokenize_vietnamese(prediction)
        ref_tokens = tokenize_vietnamese(reference)
        
        # BLEU-1
        bleu_1 = compute_bleu_1(pred_tokens, ref_tokens)
        
        # ROUGE-L
        rouge_l = compute_rouge_l(pred_tokens, ref_tokens)
        
        # Token F1
        token_f1 = compute_token_f1(pred_tokens, ref_tokens)
        
        return {
            "bleu_1": bleu_1,
            "rouge_l_precision": rouge_l["precision"],
            "rouge_l_recall": rouge_l["recall"],
            "rouge_l_f1": rouge_l["f1"],
            "token_precision": token_f1["precision"],
            "token_recall": token_f1["recall"],
            "token_f1": token_f1["f1"],
        }
    
    def evaluate_batch(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a batch and return averaged metrics.
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            
        Returns:
            Dictionary of averaged metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        if not predictions:
            return {}
        
        all_results = []
        for pred, ref in zip(predictions, references):
            result = self.evaluate_single(pred, ref)
            all_results.append(result)
        
        # Average across all samples
        avg_results = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            avg_results[f"{key}_mean"] = sum(values) / len(values)
            
            # Also compute std for key metrics
            if key in ["bleu_1", "rouge_l_f1", "token_f1"]:
                mean = avg_results[f"{key}_mean"]
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                avg_results[f"{key}_std"] = variance ** 0.5
        
        avg_results["num_samples"] = len(predictions)
        
        return avg_results
    
    def evaluate_with_details(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate batch and return both aggregate and per-sample results.
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            
        Returns:
            Tuple of (aggregate_metrics, per_sample_metrics)
        """
        per_sample = []
        for pred, ref in zip(predictions, references):
            result = self.evaluate_single(pred, ref)
            per_sample.append(result)
        
        aggregate = self.evaluate_batch(predictions, references)
        
        return aggregate, per_sample


# Example usage and testing
if __name__ == "__main__":
    evaluator = LexicalEvaluator()
    
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
            "prediction": "Điện thoại này có màn hình AMOLED 6.7 inch",
            "reference": "Sản phẩm được trang bị màn hình AMOLED kích thước 6.7 inch"
        }
    ]
    
    print("=" * 60)
    print("LEXICAL METRICS EVALUATION TEST")
    print("=" * 60)
    
    predictions = [tc["prediction"] for tc in test_cases]
    references = [tc["reference"] for tc in test_cases]
    
    # Per-sample evaluation
    for i, tc in enumerate(test_cases):
        print(f"\nSample {i+1}:")
        print(f"  Prediction: {tc['prediction'][:50]}...")
        print(f"  Reference:  {tc['reference'][:50]}...")
        
        result = evaluator.evaluate_single(tc["prediction"], tc["reference"])
        print(f"  BLEU-1: {result['bleu_1']:.4f}")
        print(f"  ROUGE-L F1: {result['rouge_l_f1']:.4f}")
        print(f"  Token F1: {result['token_f1']:.4f}")
    
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
