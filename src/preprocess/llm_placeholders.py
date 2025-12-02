"""
LLM helpers using Gemini (set GEMINI_API_KEY in .env).
- TitleLLMEnricher: brand/model/product_line/full_title_eng/category_eng
- SpecsLLMEnricher: normalize specs dict -> list[SpecItem]
"""
import json
import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv
from preprocess.models import CleanRecord, TitleLLMResult, SpecItem

# Load environment and configure Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_env_model = os.getenv("GEMINI_MODEL")
# Normalize model names; v1beta expects base ids like "gemini-1.5-flash" or "gemini-1.5-pro"
def normalize_model(name: str | None) -> str:
    if not name:
        return "gemini-1.5-flash"
    return name.replace("-latest", "")

GEMINI_MODEL = normalize_model(_env_model)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ GEMINI_API_KEY missing - LLM calls will fail.")


class TitleLLMEnricher:
    def __init__(self, model: str = GEMINI_MODEL):
        self.model = model

    def run_batch(self, records: List[CleanRecord]) -> List[TitleLLMResult]:
        results: List[TitleLLMResult] = []
        model = genai.GenerativeModel(self.model)

        for rec in records:
            if not rec.title:
                results.append(TitleLLMResult(llm_processed=False))
                continue

            prompt = (
                "Bạn là chuyên gia trích xuất thông tin sản phẩm.\n"
                "Phân tích title và category, trả JSON với các trường: "
                "brand (viết hoa chữ cái đầu), model, product_line, full_title_eng, category_eng. "
                "Giữ nguyên số/ký tự đặc biệt trong model.\n"
                f"Title: {rec.title}\nCategory: {rec.category}\n"
                'Trả về JSON object, ví dụ: {"brand": "...", "model": "...", "product_line": "...", "full_title_eng": "...", "category_eng": "..."}'
            )

            try:
                resp = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                )
                data = json.loads(resp.text or "{}")
                results.append(
                    TitleLLMResult(
                        brand=data.get("brand"),
                        model=data.get("model"),
                        product_line=data.get("product_line"),
                        full_title_eng=data.get("full_title_eng"),
                        category_eng=data.get("category_eng"),
                        llm_processed=True,
                    )
                )
            except Exception as e:
                print(f"LLM title enrich failed for '{rec.title[:40]}': {e}")
                results.append(TitleLLMResult(llm_processed=False))

        return results


class SpecsLLMEnricher:
    def __init__(self, model: str = GEMINI_MODEL):
        self.model = model

    def run(self, record: CleanRecord) -> List[SpecItem]:
        if not record.product_blob:
            return []

        model = genai.GenerativeModel(self.model)
        prompt = (
            "Bạn là chuyên gia chuẩn hóa thông số kỹ thuật sản phẩm.\n"
            "Input: JSON specs key->value (tiếng Việt).\n"
            "Output: JSON array, mỗi item có:\n"
            "- standardized_key (VI)\n"
            "- standardized_key_eng (EN)\n"
            "- standardized_value (VI)\n"
            "- standardized_value_eng (EN)\n"
            "- category (VI)\n"
            "- category_eng (EN)\n"
            "- numerical_value_list (list số, nếu không có để [])\n"
            "- unit_list (list đơn vị, nếu không có để [])\n"
            "Categories chuẩn: performance, display, memory, storage, graphics, battery, connectivity, camera, design, software, security, audio, cooling, other.\n"
            f"Specs JSON:\n{json.dumps(record.product_blob, ensure_ascii=False)}\n"
            "Chỉ trả JSON array."
        )

        try:
            resp = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            data = json.loads(resp.text or "[]")
            items = data if isinstance(data, list) else data.get("items") or data.get("specs") or []

            parsed: List[SpecItem] = []
            for item in items:
                try:
                    parsed.append(
                        SpecItem(
                            standardized_key=item.get("standardized_key", ""),
                            standardized_key_eng=item.get("standardized_key_eng", item.get("standardized_key_en", "")),
                            standardized_value=item.get("standardized_value", ""),
                            standardized_value_eng=item.get("standardized_value_eng", item.get("standardized_value_en", "")),
                            category=item.get("category", ""),
                            category_eng=item.get("category_eng", item.get("category_en", "")),
                            numerical_value_list=item.get("numerical_value_list", []) or [],
                            unit_list=item.get("unit_list", []) or [],
                        )
                    )
                except Exception:
                    continue
            return parsed
        except Exception as e:
            print(f"LLM specs enrich failed: {e}")
            return []
