"""Tests for MinerUParser — scanned PDF parsing via MinerU Flash mode."""
import pytest

mineru_available = False
try:
    from langchain_mineru import MinerULoader  # noqa: F401

    mineru_available = True
except ImportError:
    pass


@pytest.mark.skipif(not mineru_available, reason="MinerU not installed")
class TestMinerUParser:
    def test_parse_scanned_pdf_returns_elements(self, tmp_path):
        """MinerU parses a scanned PDF into StructuredElements."""
        import shutil

        fixture = "tests/fixtures/scanned_sample.pdf"
        dest = tmp_path / "scanned.pdf"
        shutil.copy(fixture, str(dest))

        from backend.services.parsing.mineru_parser import MinerUParser

        parser = MinerUParser()
        elements = parser.parse(str(dest))
        assert len(elements) > 0
        paragraphs = [e for e in elements if e.element_type == "paragraph"]
        assert len(paragraphs) > 0

    def test_parse_returns_structured_elements(self, tmp_path):
        """返回的元素应符合 StructuredElement 结构。"""
        import shutil

        fixture = "tests/fixtures/scanned_sample.pdf"
        dest = tmp_path / "scanned2.pdf"
        shutil.copy(fixture, str(dest))

        from backend.services.parsing.mineru_parser import MinerUParser

        parser = MinerUParser()
        elements = parser.parse(str(dest))
        for e in elements:
            assert hasattr(e, "content")
            assert hasattr(e, "element_type")
            assert e.element_type in ("heading", "paragraph", "table", "code", "list")

    def test_file_not_found_raises(self):
        from backend.services.parsing.mineru_parser import MinerUParser

        parser = MinerUParser()
        with pytest.raises((FileNotFoundError, Exception)):
            parser.parse("/nonexistent/path.pdf")


@pytest.mark.skipif(mineru_available, reason="MinerU is installed")
class TestMinerUParserNotInstalled:
    def test_import_raises_importerror(self):
        """MinerU 未安装时，实例化并调用应抛 ImportError。"""
        with pytest.raises(ImportError):
            from backend.services.parsing.mineru_parser import MinerUParser

            p = MinerUParser()
            p.parse("dummy.pdf")
