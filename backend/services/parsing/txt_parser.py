from backend.services.parsing.base import BaseParser, StructuredElement


class TxtParser(BaseParser):
    def parse(self, filepath: str) -> list[StructuredElement]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self.parse_string(text)

    def parse_string(self, text: str) -> list[StructuredElement]:
        result = []
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                result.append(
                    StructuredElement(content=para, element_type="paragraph")
                )
        return result
