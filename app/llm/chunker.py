from typing import List

class TextChunker:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough token estimation.
        Approximately 1 token ≈ 4 characters for English text.
        """
        return max(1, len(text) // 4)

    def split_large_page(self, page: str) -> List[str]:
        """
        Split a page into smaller chunks using paragraph boundaries.
        """
        paragraphs = page.split("\n\n")

        chunks = []
        current = ""

        for para in paragraphs:

            candidate = para if not current else current + "\n\n" + para

            if self.estimate_tokens(candidate) <= self.max_tokens:
                current = candidate

            else:
                if current:
                    chunks.append(current.strip())

                # If one paragraph is too large,
                # split by lines.
                if self.estimate_tokens(para) > self.max_tokens:

                    lines = para.split("\n")
                    current = ""

                    for line in lines:

                        candidate = (
                            line
                            if not current
                            else current + "\n" + line
                        )

                        if self.estimate_tokens(candidate) <= self.max_tokens:
                            current = candidate
                        else:
                            if current:
                                chunks.append(current.strip())
                            current = line

                else:
                    current = para

        if current:
            chunks.append(current.strip())

        return chunks

    def chunk(self, pages: List[str]) -> List[str]:
        """
        Build chunks using complete pages whenever possible.
        If a page is too large, split it into paragraphs.
        """
        chunks = []

        current = ""

        for page in pages:

            # Oversized page
            if self.estimate_tokens(page) > self.max_tokens:

                if current:
                    chunks.append(current.strip())
                    current = ""

                chunks.extend(
                    self.split_large_page(page)
                )

                continue

            candidate = (
                page
                if not current
                else current + "\n\n" + page
            )

            if self.estimate_tokens(candidate) <= self.max_tokens:
                current = candidate

            else:
                if current:
                    chunks.append(current.strip())

                current = page

        if current:
            chunks.append(current.strip())

        return chunks