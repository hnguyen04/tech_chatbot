"""
Data loader for QA evaluation pairs.

Expected JSON format:
[
    {
        "question": "...",
        "answer": "...",
        "relevant_product_ids": [123, 456]  # Optional but recommended
    },
    ...
]
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any


@dataclass
class QASample:
    """Single QA evaluation sample."""
    question: str
    answer: str
    relevant_product_ids: List[int] = field(default_factory=list)
    query_type: str = "unknown"  # simple, comparison, constraint, recommendation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "relevant_product_ids": self.relevant_product_ids,
            "query_type": self.query_type
        }


@dataclass
class EvalPrediction:
    """Prediction result for evaluation."""
    query: str
    ground_truth: str
    prediction: str
    context: str = ""
    retrieved_ids: List[int] = field(default_factory=list)
    reranked_ids: List[int] = field(default_factory=list)
    relevant_ids: List[int] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    # Agentic-specific
    tool_log: List[str] = field(default_factory=list)
    iterations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "prediction": self.prediction,
            "context": self.context,
            "retrieved_ids": self.retrieved_ids,
            "reranked_ids": self.reranked_ids,
            "relevant_ids": self.relevant_ids,
            "sources": self.sources,
            "tool_log": self.tool_log,
            "iterations": self.iterations
        }


def classify_query_type(question: str) -> str:
    """
    Classify query type based on keywords.
    
    Types:
    - simple: Direct lookup (price, specs)
    - comparison: Compare products
    - constraint: Budget or feature constraints
    - recommendation: Ask for suggestions
    """
    question_lower = question.lower()
    
    # Comparison patterns
    comparison_patterns = [
        r"so sánh", r"khác.*gì", r"hơn.*không", r"tốt hơn",
        r"nên.*hay", r"chọn.*hay", r"vs", r"versus"
    ]
    for pattern in comparison_patterns:
        if re.search(pattern, question_lower):
            return "comparison"
    
    # Constraint patterns (budget, features)
    constraint_patterns = [
        r"dưới.*triệu", r"trên.*triệu", r"khoảng.*triệu",
        r"tầm.*triệu", r"giá.*rẻ", r"budget", r"ngân sách",
        r"có.*tính năng", r"hỗ trợ.*không"
    ]
    for pattern in constraint_patterns:
        if re.search(pattern, question_lower):
            return "constraint"
    
    # Recommendation patterns
    recommendation_patterns = [
        r"nên mua", r"gợi ý", r"đề xuất", r"recommend",
        r"tư vấn", r"chọn.*nào", r"mua.*gì", r"top.*điện thoại"
    ]
    for pattern in recommendation_patterns:
        if re.search(pattern, question_lower):
            return "recommendation"
    
    # Default: simple lookup
    return "simple"


def load_qa_data(filepath: str) -> List[QASample]:
    """
    Load QA pairs from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of QASample objects
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {filepath}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        # Validate required fields
        if "question" not in item:
            print(f"Warning: Skipping item without 'question' field")
            continue
        
        question = item.get("question", "")
        answer = item.get("answer", "")
        relevant_ids = item.get("relevant_product_ids", [])
        
        # Ensure relevant_ids are integers
        if relevant_ids:
            relevant_ids = [int(x) for x in relevant_ids]
        
        # Auto-classify query type
        query_type = item.get("query_type") or classify_query_type(question)
        
        sample = QASample(
            question=question,
            answer=answer,
            relevant_product_ids=relevant_ids,
            query_type=query_type
        )
        samples.append(sample)
    
    print(f"Loaded {len(samples)} QA samples from {filepath}")
    return samples


def split_by_type(samples: List[QASample]) -> Dict[str, List[QASample]]:
    """
    Split samples by query type.
    
    Args:
        samples: List of QA samples
        
    Returns:
        Dictionary mapping query_type to list of samples
    """
    by_type: Dict[str, List[QASample]] = {
        "simple": [],
        "comparison": [],
        "constraint": [],
        "recommendation": [],
        "unknown": []
    }
    
    for sample in samples:
        qtype = sample.query_type
        if qtype in by_type:
            by_type[qtype].append(sample)
        else:
            by_type["unknown"].append(sample)
    
    # Print distribution
    print("\nQuery type distribution:")
    for qtype, items in by_type.items():
        if items:
            print(f"  {qtype}: {len(items)} ({100*len(items)/len(samples):.1f}%)")
    
    return by_type


def validate_qa_data(samples: List[QASample]) -> Dict[str, Any]:
    """
    Validate QA data quality.
    
    Args:
        samples: List of QA samples
        
    Returns:
        Validation report dictionary
    """
    report = {
        "total_samples": len(samples),
        "has_answer": 0,
        "has_product_ids": 0,
        "avg_answer_length": 0,
        "avg_question_length": 0,
        "issues": []
    }
    
    total_answer_len = 0
    total_question_len = 0
    
    for i, sample in enumerate(samples):
        if sample.answer:
            report["has_answer"] += 1
            total_answer_len += len(sample.answer)
        else:
            report["issues"].append(f"Sample {i}: Missing answer")
        
        if sample.relevant_product_ids:
            report["has_product_ids"] += 1
        
        total_question_len += len(sample.question)
    
    if samples:
        report["avg_answer_length"] = total_answer_len / len(samples)
        report["avg_question_length"] = total_question_len / len(samples)
    
    # Summary
    print("\nQA Data Validation:")
    print(f"  Total samples: {report['total_samples']}")
    print(f"  With answers: {report['has_answer']} ({100*report['has_answer']/len(samples):.1f}%)")
    print(f"  With product IDs: {report['has_product_ids']} ({100*report['has_product_ids']/len(samples):.1f}%)")
    print(f"  Avg question length: {report['avg_question_length']:.0f} chars")
    print(f"  Avg answer length: {report['avg_answer_length']:.0f} chars")
    
    if report["issues"]:
        print(f"  Issues found: {len(report['issues'])}")
    
    return report


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "data/qa_pairs.json"
    
    try:
        samples = load_qa_data(filepath)
        validate_qa_data(samples)
        split_by_type(samples)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide a valid QA file path")
