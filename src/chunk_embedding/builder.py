class SpecBlockBuilder:
    """
    Build 1 semantic spec block cho 1 product
    """

    def build(self, specs: list[dict]) -> str | None:
        if not specs:
            return None

        lines = ["Specifications:"]

        for s in specs:
            key = s.get("standardized_key_eng") or s.get("standardized_key")
            if not key:
                continue

            val = s.get("standardized_value_eng") or s.get("standardized_value") or ""
            cat = (s.get("category_eng") or s.get("category") or "").lower()

            # numerical + unit override
            if s.get("numerical_value_list"):
                nums = ", ".join(map(str, s["numerical_value_list"]))
                units = ", ".join(s.get("unit_list") or [])
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
        parts = []

        # ===== PRODUCT IDENTITY (VERY IMPORTANT FOR EMBEDDING) =====
        parts.append(f"Title: {record['full_title']}")

        if record.get("brand"):
            parts.append(f"Brand: {record['brand']}")

        if record.get("model"):
            parts.append(f"Model: {record['model']}")

        if record.get("product_line"):
            parts.append(f"Product line: {record['product_line']}")

        if record.get("category") or record.get("category_eng"):
            parts.append(
                f"Category: {record.get('category') or record.get('category_eng')}"
            )

        if record.get("price"):
            parts.append(f"Price: {record['price']} VNĐ")

        # separator để embedding “hiểu” section
        parts.append("")

        # ===== STRUCTURED SPECS =====
        if record.get("content_type") == "product" and specs_block:
            parts.append(specs_block)
            parts.append("")

        # ===== UNSTRUCTURED CONTENT =====
        parts.append("Description:")
        parts.append(chunk_text)

        return "\n".join(parts)
