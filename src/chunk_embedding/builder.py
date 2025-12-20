class SpecBlockBuilder:
    """
    Build 1 semantic spec block cho 1 product
    """

    def build(self, specs: list[dict]) -> str | None:
        if not specs:
            return None

        lines = ["Specifications:"]

        for s in specs:
            key = s["standardized_key_eng"] or s["standardized_key"]
            val = s["standardized_value_eng"] or s["standardized_value"]
            cat = (s["category_eng"] or s["category"] or "").lower()

            # numerical + unit override
            if s["numerical_value_list"]:
                nums = ", ".join(map(str, s["numerical_value_list"]))
                units = ", ".join(s["unit_list"] or [])
                val = f"{nums} {units}".strip()

            lines.append(f"- {key}: {val} ({cat})")

        return "\n".join(lines)

class DocumentBuilder:
    def build(
        self,
        record: dict,
        specs_block: str | None,
        chunk_text: str,
    ) -> str:
        parts = [
            f"Title: {record['full_title']}",
            f"Category: {record['category_eng'] or record['category']}",
        ]

        if record.get("brand"):
            parts.append(f"Brand: {record['brand']}")

        if record["content_type"] == "product" and specs_block:
            parts.append(specs_block)

        parts.append("Content:")
        parts.append(chunk_text)

        return "\n".join(parts)
