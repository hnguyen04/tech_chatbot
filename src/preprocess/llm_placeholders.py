"""
LLM helpers using Gemini (set GEMINI_API_KEY in .env).
- TitleLLMEnricher: brand/model/product_line/full_title_eng/category_eng (batch với throttling)
- SpecsLLMEnricher: normalize specs dict -> list[SpecItem]
"""
import json
import os
import time
from typing import List, Dict
from llm.factory import build_llm

import google.generativeai as genai
from dotenv import load_dotenv
from preprocess.models import CleanRecord, TitleLLMResult, SpecItem

# Load environment and configure Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_env_model = os.getenv("GEMINI_MODEL")


def normalize_model(name: str | None) -> str:
    if not name:
        return "gemini-1.5-flash"
    return name.replace("-latest", "")


GEMINI_MODEL = normalize_model(_env_model)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print(" GEMINI_API_KEY missing - LLM calls will fail.")

# LLM batch/throttle configs
TITLE_BATCH_SIZE = int(os.getenv("TITLE_LLM_BATCH", "10"))
TITLE_RETRIES = int(os.getenv("TITLE_LLM_RETRIES", "2"))
TITLE_SLEEP = float(os.getenv("TITLE_LLM_SLEEP", "1.0"))  # seconds between calls
SPECS_RETRIES = int(os.getenv("SPECS_LLM_RETRIES", "2"))
SPECS_SLEEP = float(os.getenv("SPECS_LLM_SLEEP", "1.0"))


class TitleLLMEnricher:
    def __init__(self, model: str = GEMINI_MODEL):
        # 🔧 FIX: build_llm trả về callable llm(prompt)
        self.llm = build_llm()

    def run_batch(self, records: List[CleanRecord]) -> List[TitleLLMResult]:
        results: List[TitleLLMResult] = []

        # 🔧 FIX: bỏ genai.GenerativeModel (không dùng nữa)
        for i in range(0, len(records), TITLE_BATCH_SIZE):
            chunk = records[i : i + TITLE_BATCH_SIZE]
            user_lines = []

            for idx, rec in enumerate(chunk):
                user_lines.append(
                    f"{idx+1}. Title: {rec.title}\nCategory: {rec.category}"
                )

            prompt = (
                "Bạn là chuyên gia trích xuất thông tin sản phẩm.\n"
                "Nhận danh sách (title, category) và trả JSON array (theo đúng thứ tự input), mỗi item gồm: "
                "brand (viết hoa chữ cái đầu), model, product_line, full_title_eng, category_eng. "
                "Giữ nguyên số/ký tự đặc biệt trong model.\n"
                "Input:\n"
                + "\n".join(user_lines)
                + '\nOutput: JSON array, ví dụ: '
                    '[{"brand": "...", "model": "...", "product_line": "...", '
                    '"full_title_eng": "...", "category_eng": "..."}]'
            )

            # 🔧 FIX: truyền self.llm thay vì model
            parsed_items = self._call_with_retry(
                prompt,
                expected=len(chunk),
            )

            for rec, item in zip(chunk, parsed_items):
                if not item:
                    results.append(TitleLLMResult(llm_processed=False))
                    continue

                results.append(
                    TitleLLMResult(
                        brand=item.get("brand"),
                        model=item.get("model"),
                        product_line=item.get("product_line"),
                        full_title_eng=item.get("full_title_eng"),
                        category_eng=item.get("category_eng"),
                        llm_processed=True,
                    )
                )

            if TITLE_SLEEP:
                time.sleep(TITLE_SLEEP)

        return results

    def _call_with_retry(self, prompt: str, expected: int) -> List[dict]:
        last_err = None

        for attempt in range(TITLE_RETRIES + 1):
            try:
                # 🔧 FIX: gọi self.llm(prompt)
                items = self.llm(prompt)

                # 🔧 FIX: build_llm đã parse JSON → không json.loads
                if not isinstance(items, list):
                    items = []

                while len(items) < expected:
                    items.append({})

                return items[:expected]

            except Exception as e:
                last_err = e
                print(
                    f"LLM title batch failed "
                    f"(attempt {attempt+1}/{TITLE_RETRIES+1}): {e}"
                )
                time.sleep((attempt + 1) * TITLE_SLEEP)

        return [{} for _ in range(expected)]


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
            "Categories chuẩn tiếng Anh: performance, display, memory, storage, graphics, battery, connectivity, camera, design, software, security, audio, cooling, other.\n Có thể đưa thêm category tiếng Việt tương ứng và category khác nếu cần thiết.\n"
            f"Specs JSON:\n{json.dumps(record.product_blob, ensure_ascii=False)}\n"
            "Chỉ trả JSON array."
        )

        items = self._call_with_retry(model, prompt)
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

    def _call_with_retry(self, model, prompt: str) -> List[dict]:
        last_err = None
        for attempt in range(SPECS_RETRIES + 1):
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                )
                data = json.loads(resp.text or "[]")
                items = data if isinstance(data, list) else data.get("items") or data.get("specs") or []
                if not isinstance(items, list):
                    items = []
                return items
            except Exception as e:
                last_err = e
                print(f"LLM specs enrich failed (attempt {attempt+1}/{SPECS_RETRIES+1}): {e}")
                time.sleep((attempt + 1) * SPECS_SLEEP)
        return []
    
class SpecKeyEnricher:
    def __init__(self, batch_size: int = 20):
        self.llm = build_llm()
        self.batch_size = batch_size

    def _build_prompt(self, rows: List[Dict]) -> str:
        """
        rows: [{id, raw_key, normalized_key}]
        """
        items = "\n".join(
            f"- id={r['id']}, key=\"{r['raw_key']}\""
            for r in rows
        )

        return f"""
            You are normalizing product specification keys.

            For each input key, return:
            - standardized_key_eng: snake_case English key
            - category_vi: Vietnamese category name
            - category_eng: English category name

            Rules:
            - standardized_key_eng MUST be concise, lowercase, snake_case
            - category_vi should be human-readable (Vietnamese)
            - category_eng should be Title Case English
            - DO NOT invent information
            - DO NOT explain
            - OUTPUT JSON ONLY

            Input keys:
            {items}

            Output format (JSON array):
            [
            {{
                "id": <number>,
                "standardized_key_eng": "...",
                "category_vi": "...",
                "category_eng": "..."
            }}
            ]
            """.strip()

    def enrich_batch(self, rows: List[Dict]) -> List[Dict]:
        prompt = self._build_prompt(rows)
        result = self.llm(prompt)

        if not isinstance(result, list):
            raise ValueError("LLM did not return JSON array")

        return result
