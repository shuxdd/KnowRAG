import logging
import os

from backend.services.parsing.base import BaseParser, StructuredElement

logger = logging.getLogger(__name__)


class MinerUParser(BaseParser):
    """MinerU-based PDF parser for scanned documents and complex layouts.

    Uses MinerU Flash mode (free, no token required) to parse PDFs into Markdown,
    then reuses MarkdownParser to convert to StructuredElement list.

    Flash mode limits: <= 10 MB per file, <= 20 pages.
    """

    def parse(self, filepath: str) -> list[StructuredElement]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        from langchain_mineru import MinerULoader

        file_size = os.path.getsize(filepath)
        if file_size > 10 * 1024 * 1024:
            logger.warning(
                "File %s exceeds 10 MB, MinerU Flash mode may fail", filepath
            )

        loader = MinerULoader(source=filepath, mode="flash")
        docs = loader.load()

        if not docs:
            logger.warning("MinerU returned no content for %s", filepath)
            return []

        md_content = "\n\n".join(doc.page_content for doc in docs)

        from backend.services.parsing.markdown_parser import MarkdownParser

        return MarkdownParser().parse_string(md_content)
