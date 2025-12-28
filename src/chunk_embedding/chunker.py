# chunk_embedding/chunker.py

from typing import List, Tuple
import re


class Chunker:
    def __init__(
        self,
        max_chars: int = 4096,
        overlap_units: int = 1,
    ):
        self.max_chars = max_chars
        self.overlap_units = overlap_units

        # spec: bắt đầu bằng "-"
        self.spec_line = re.compile(r'^\s*-\s+')

        # sentence boundary:
        # - dấu , luôn được
        # - . ! ? … nhưng . không đứng sau số
        self.sentence_splitter = re.compile(
            r'(?<=[,!…!?])\s+|(?<!\d)\.\s+|\n+'
        )

    def _split_units(self, text: str) -> List[str]:
        """
        Trả về list các unit:
        - mỗi spec là 1 unit
        - mỗi câu là 1 unit
        """
        lines = text.splitlines()
        units: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # spec line → giữ nguyên
            if self.spec_line.match(line):
                units.append(line)
                continue

            # sentence line → tách tiếp
            parts = self.sentence_splitter.split(line)
            for p in parts:
                p = p.strip()
                if p:
                    units.append(p)

        return units

    def chunk(self, text: str | None) -> List[Tuple[int, str]]:
        """
        Chunk theo unit (spec / sentence).
        Đảm bảo đầu chunk luôn là spec hoặc sentence hợp lệ.
        """
        if not text or not text.strip():
            return [(0, "")]

        units = self._split_units(text)
        if not units:
            return [(0, text.strip())]

        chunks: List[Tuple[int, str]] = []

        current: List[str] = []
        current_len = 0
        idx = 0

        for unit in units:
            unit_len = len(unit)

            # unit quá dài → hard cut (rất hiếm)
            if unit_len > self.max_chars:
                if current:
                    chunks.append((idx, " ".join(current)))
                    idx += 1
                    current = []
                    current_len = 0

                for i in range(0, unit_len, self.max_chars):
                    part = unit[i:i + self.max_chars].strip()
                    if part:
                        chunks.append((idx, part))
                        idx += 1
                continue

            # nếu vượt max_chars → flush
            if current_len + unit_len > self.max_chars:
                chunks.append((idx, " ".join(current)))
                idx += 1

                # overlap theo unit
                if self.overlap_units > 0:
                    overlap = current[-self.overlap_units:]
                    current = overlap + [unit]
                    current_len = sum(len(x) for x in current)
                else:
                    current = [unit]
                    current_len = unit_len
            else:
                current.append(unit)
                current_len += unit_len

        if current:
            chunks.append((idx, " ".join(current)))

        if not chunks:
            return [(0, "")]

        return chunks
