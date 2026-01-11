"""
LLM-as-Judge evaluation - Budget optimized.

Uses Gemini Flash for cost efficiency.
Evaluates only a sample of predictions to save API costs.
"""
import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import google.generativeai as genai

from src.rag.config import GEMINI_API_KEY, GEMINI_MODEL


# Vietnamese evaluation prompt for better assessment
JUDGE_PROMPT_VI = """Bạn là một chuyên gia đánh giá chất lượng câu trả lời của hệ thống RAG (Retrieval-Augmented Generation).

**Câu hỏi của người dùng:**
{query}

**Ngữ cảnh được truy xuất (Context):**
{context}

**Câu trả lời của hệ thống:**
{answer}

**Câu trả lời mẫu (Ground Truth):**
{ground_truth}

Hãy đánh giá câu trả lời của hệ thống theo 3 tiêu chí sau (thang điểm 1-5):

### 1. Faithfulness (Độ trung thực)
Câu trả lời có đúng với thông tin trong Context không? Có bịa đặt thông tin không?
- **5**: Hoàn toàn chính xác, tất cả thông tin đều có trong context
- **4**: Phần lớn chính xác, chỉ có rất ít suy luận hợp lý
- **3**: Đúng phần lớn, có vài chi tiết không có trong context
- **2**: Nhiều thông tin không có trong context hoặc sai
- **1**: Bịa đặt nhiều thông tin (hallucination nghiêm trọng)

### 2. Relevance (Độ liên quan)
Câu trả lời có trả lời đúng câu hỏi được đặt ra không?
- **5**: Trả lời đầy đủ, đúng trọng tâm câu hỏi
- **4**: Trả lời tốt, có thể thiếu một vài chi tiết nhỏ
- **3**: Trả lời được phần chính, thiếu một số thông tin quan trọng
- **2**: Trả lời sơ sài hoặc lạc đề một phần
- **1**: Hoàn toàn lạc đề hoặc không trả lời câu hỏi

### 3. Completeness (Độ đầy đủ)
Câu trả lời có bao quát hết các khía cạnh của câu hỏi không?
- **5**: Đầy đủ, chi tiết, bao quát mọi khía cạnh được hỏi
- **4**: Khá đầy đủ, có thể thiếu một vài chi tiết phụ
- **3**: Bao quát ý chính nhưng thiếu chi tiết quan trọng
- **2**: Thiếu nhiều thông tin cần thiết
- **1**: Quá sơ sài, không đủ thông tin

Trả về JSON với format sau (KHÔNG thêm bất kỳ text nào khác):
{{
  "faithfulness": {{"score": <1-5>, "reason": "<lý do ngắn gọn bằng tiếng Việt>"}},
  "relevance": {{"score": <1-5>, "reason": "<lý do ngắn gọn bằng tiếng Việt>"}},
  "completeness": {{"score": <1-5>, "reason": "<lý do ngắn gọn bằng tiếng Việt>"}}
}}"""


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    faithfulness_score: float
    faithfulness_reason: str
    relevance_score: float
    relevance_reason: str
    completeness_score: float
    completeness_reason: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": {
                "score": self.faithfulness_score,
                "reason": self.faithfulness_reason
            },
            "relevance": {
                "score": self.relevance_score,
                "reason": self.relevance_reason
            },
            "completeness": {
                "score": self.completeness_score,
                "reason": self.completeness_reason
            },
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "JudgeResult":
        return cls(
            faithfulness_score=data.get("faithfulness", {}).get("score", 0),
            faithfulness_reason=data.get("faithfulness", {}).get("reason", ""),
            relevance_score=data.get("relevance", {}).get("score", 0),
            relevance_reason=data.get("relevance", {}).get("reason", ""),
            completeness_score=data.get("completeness", {}).get("score", 0),
            completeness_reason=data.get("completeness", {}).get("reason", ""),
            error=data.get("error")
        )


class LLMJudge:
    """
    Budget-optimized LLM-as-Judge evaluator.
    
    Uses Gemini Flash for cost efficiency.
    Samples a subset of predictions to save API costs.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash"  # Cheaper than default
    ):
        """
        Initialize LLM Judge.
        
        Args:
            api_key: Gemini API key (uses env var if None)
            model_name: Model to use (flash is cheaper)
        """
        api_key = api_key or GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = {
            "temperature": 0.1,  # Low temp for consistency
            "response_mime_type": "application/json"
        }
        
        print(f"LLM Judge initialized with {model_name}")
    
    def evaluate_single(
        self,
        query: str,
        answer: str,
        context: str = "",
        ground_truth: str = ""
    ) -> JudgeResult:
        """
        Evaluate a single response.
        
        Args:
            query: User question
            answer: Generated answer
            context: Retrieved context (truncated if too long)
            ground_truth: Expected answer (if available)
            
        Returns:
            JudgeResult with scores and reasons
        """
        # Truncate inputs to save tokens and cost
        context_truncated = context[:2500] if context else "Không có context"
        answer_truncated = answer[:1500] if answer else "Không có câu trả lời"
        ground_truth_text = ground_truth[:1000] if ground_truth else "Không có"
        
        prompt = JUDGE_PROMPT_VI.format(
            query=query,
            context=context_truncated,
            answer=answer_truncated,
            ground_truth=ground_truth_text
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            result_data = json.loads(response.text)
            return JudgeResult.from_dict(result_data)
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return JudgeResult(
                faithfulness_score=0,
                faithfulness_reason="Parse error",
                relevance_score=0,
                relevance_reason="Parse error",
                completeness_score=0,
                completeness_reason="Parse error",
                error=f"JSON parse error: {str(e)}"
            )
        except Exception as e:
            print(f"LLM Judge error: {e}")
            return JudgeResult(
                faithfulness_score=0,
                faithfulness_reason="API error",
                relevance_score=0,
                relevance_reason="API error",
                completeness_score=0,
                completeness_reason="API error",
                error=str(e)
            )
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, str]],
        sample_size: Optional[int] = 50,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of samples.
        
        Samples a subset to save API costs.
        
        Args:
            samples: List of dicts with keys: query, answer, context, ground_truth
            sample_size: Number of samples to evaluate (None for all)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with aggregate scores and details
        """
        # Random sample for cost efficiency
        if sample_size and len(samples) > sample_size:
            random.seed(random_seed)
            samples = random.sample(samples, sample_size)
            print(f"Sampling {sample_size} out of {len(samples)} for LLM evaluation")
        
        print(f"Evaluating {len(samples)} samples with LLM Judge...")
        
        all_results: List[JudgeResult] = []
        errors = 0
        
        for i, sample in enumerate(samples):
            print(f"  Evaluating {i+1}/{len(samples)}...", end="\r")
            
            result = self.evaluate_single(
                query=sample.get("query", ""),
                answer=sample.get("answer", sample.get("prediction", "")),
                context=sample.get("context", ""),
                ground_truth=sample.get("ground_truth", "")
            )
            
            all_results.append(result)
            
            if result.error:
                errors += 1
        
        print(f"  Completed {len(samples)} evaluations ({errors} errors)")
        
        # Aggregate scores (exclude errors)
        valid_results = [r for r in all_results if not r.error]
        
        if not valid_results:
            return {
                "error": "All evaluations failed",
                "num_evaluated": 0,
                "num_errors": errors
            }
        
        faithfulness_scores = [r.faithfulness_score for r in valid_results]
        relevance_scores = [r.relevance_score for r in valid_results]
        completeness_scores = [r.completeness_score for r in valid_results]
        
        def compute_stats(scores: List[float]) -> Dict[str, float]:
            if not scores:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            return {
                "mean": mean,
                "std": variance ** 0.5,
                "min": min(scores),
                "max": max(scores)
            }
        
        return {
            "faithfulness": compute_stats(faithfulness_scores),
            "relevance": compute_stats(relevance_scores),
            "completeness": compute_stats(completeness_scores),
            "overall_mean": (
                sum(faithfulness_scores) + sum(relevance_scores) + sum(completeness_scores)
            ) / (3 * len(valid_results)),
            "num_evaluated": len(valid_results),
            "num_errors": errors,
            "per_sample_results": [r.to_dict() for r in all_results]
        }
    
    def evaluate_with_budget(
        self,
        samples: List[Dict[str, str]],
        max_cost_usd: float = 0.50,
        cost_per_sample: float = 0.01  # Rough estimate for Gemini Flash
    ) -> Dict[str, Any]:
        """
        Evaluate with budget constraint.
        
        Args:
            samples: List of evaluation samples
            max_cost_usd: Maximum budget in USD
            cost_per_sample: Estimated cost per API call
            
        Returns:
            Evaluation results
        """
        max_samples = int(max_cost_usd / cost_per_sample)
        sample_size = min(max_samples, len(samples))
        
        print(f"Budget: ${max_cost_usd:.2f} -> Evaluating {sample_size} samples")
        
        return self.evaluate_batch(samples, sample_size=sample_size)


def format_judge_results_for_report(results: Dict[str, Any]) -> str:
    """
    Format LLM Judge results as markdown for report.
    
    Args:
        results: Results from evaluate_batch
        
    Returns:
        Markdown formatted string
    """
    if "error" in results and results.get("num_evaluated", 0) == 0:
        return f"**Error**: {results['error']}"
    
    lines = [
        "### LLM-as-Judge Evaluation Results\n",
        f"**Samples evaluated**: {results['num_evaluated']}",
        f"**Errors**: {results['num_errors']}\n",
        "| Dimension | Mean | Std | Min | Max |",
        "|-----------|------|-----|-----|-----|",
    ]
    
    for dim in ["faithfulness", "relevance", "completeness"]:
        stats = results.get(dim, {})
        lines.append(
            f"| {dim.capitalize()} | "
            f"{stats.get('mean', 0):.2f} | "
            f"{stats.get('std', 0):.2f} | "
            f"{stats.get('min', 0):.1f} | "
            f"{stats.get('max', 0):.1f} |"
        )
    
    lines.append(f"\n**Overall Mean Score**: {results.get('overall_mean', 0):.2f}/5")
    
    return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    # Test with mock data (actual usage requires API key)
    print("=" * 60)
    print("LLM JUDGE TEST")
    print("=" * 60)
    
    try:
        judge = LLMJudge()
        
        # Single evaluation test
        result = judge.evaluate_single(
            query="iPhone 15 Pro Max giá bao nhiêu?",
            answer="iPhone 15 Pro Max có giá 28.990.000 VNĐ tại Thế Giới Di Động. Sản phẩm được trang bị chip A17 Pro và camera 48MP.",
            context="iPhone 15 Pro Max 256GB - Giá: 28,990,000đ. Chip: Apple A17 Pro. Camera: 48MP + 12MP + 12MP.",
            ground_truth="Giá iPhone 15 Pro Max là 28.990.000 đồng"
        )
        
        print("\nSingle Evaluation Result:")
        print(f"  Faithfulness: {result.faithfulness_score}/5 - {result.faithfulness_reason}")
        print(f"  Relevance: {result.relevance_score}/5 - {result.relevance_reason}")
        print(f"  Completeness: {result.completeness_score}/5 - {result.completeness_reason}")
        
    except ValueError as e:
        print(f"Skipping test - API key not configured: {e}")
    except Exception as e:
        print(f"Test error: {e}")
