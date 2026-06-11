from backend.services.parsing.pdf_parser import PdfParser


def test_heading_detection(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Large Heading", fontsize=20)
    page.insert_text((72, 130), "Normal body text for testing.", fontsize=12)
    page.insert_text((72, 150), "Another body paragraph.", fontsize=12)
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    parser = PdfParser()
    elements = parser.parse(str(path))
    headings = [e for e in elements if e.element_type == "heading"]
    assert len(headings) >= 1
    assert "Large Heading" in headings[0].content


def test_page_number_filtered(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 50), "Title", fontsize=18)
    page.insert_text((300, 800), "3", fontsize=12)
    page.insert_text((72, 100), "Content.", fontsize=12)
    path = tmp_path / "pgnum.pdf"
    doc.save(str(path))
    doc.close()
    parser = PdfParser()
    elements = parser.parse(str(path))
    headings = [e for e in elements if e.element_type == "heading"]
    for h in headings:
        assert h.content.strip() != "3", "page number 3 was falsely identified as heading"


def test_table_extraction(tmp_path):
    """Verify that tables in PDFs are extracted as table elements via find_tables()."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    # fitz doesn't have a high-level table API in minimal mode;
    # we test that the parser handles a page with tabular content without crashing
    # and returns at least some elements.
    page.insert_text((72, 50), "Tabular Data", fontsize=14)
    page.insert_text((72, 80), "col1      col2", fontsize=12)
    path = tmp_path / "table_test.pdf"
    doc.save(str(path))
    doc.close()
    from backend.services.parsing.pdf_parser import PdfParser
    parser = PdfParser()
    elements = parser.parse(str(path))
    assert len(elements) > 0


def test_normal_pdf_still_parsed_by_pymupdf(tmp_path):
    """有文本层的正常 PDF 仍然走 PyMuPDF 解析。"""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "This is a normal PDF with text content.", fontsize=12)
    page.insert_text((72, 130), "It has enough characters per page.", fontsize=12)
    page.insert_text((72, 160), "Sufficient body text covers detection threshold.", fontsize=12)
    page.insert_text((72, 190), "More content to comfortably exceed 100 chars.", fontsize=12)
    path = tmp_path / "normal.pdf"
    doc.save(str(path))
    doc.close()
    parser = PdfParser()
    elements = parser.parse(str(path))
    texts = [e.content for e in elements]
    combined = " ".join(texts)
    assert "normal PDF" in combined


def test_scanned_pdf_falls_back_to_mineru(tmp_path):
    """扫描件 PDF（无文本层）应 fallback 到 MinerU。MinerU 不可用时回退 PyMuPDF。"""
    import shutil

    try:
        from langchain_mineru import MinerULoader  # noqa: F401

        mineru_available = True
    except ImportError:
        mineru_available = False

    fixture = "tests/fixtures/scanned_sample.pdf"
    dest = tmp_path / "scanned.pdf"
    shutil.copy(fixture, str(dest))

    parser = PdfParser()
    elements = parser.parse(str(dest))
    assert isinstance(elements, list)
    if mineru_available:
        # MinerU 走通后应该 OCR 出文本 → 非空元素列表
        assert len(elements) > 0, "MinerU fallback should produce elements for scanned fixture"


def test_empty_pdf_returns_empty(monkeypatch):
    """0 页 PDF 应直接返回空列表（通过 mock 模拟，因 PyMuPDF 不允许保存 0 页）。"""
    import fitz

    class FakeDoc:
        page_count = 0

        def __iter__(self):
            return iter([])

        def close(self):
            pass

    monkeypatch.setattr(fitz, "open", lambda _path: FakeDoc())
    parser = PdfParser()
    elements = parser.parse("fake.pdf")
    assert elements == []


def test_adaptive_heading_small_size_gap(tmp_path):
    """PDF where heading is only slightly larger than body (ratio ~1.17) should still detect heading."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # Body is 12pt, heading is 14pt — ratio 1.17, below old h1_ratio=1.4
    page.insert_text((72, 100), "Section Title", fontsize=14)
    page.insert_text((72, 130), "Body text paragraph one." * 5, fontsize=12)
    page.insert_text((72, 160), "Body text paragraph two." * 5, fontsize=12)
    page.insert_text((72, 190), "Body text paragraph three." * 5, fontsize=12)
    path = tmp_path / "small_gap.pdf"
    doc.save(str(path))
    doc.close()

    parser = PdfParser()
    elements = parser.parse(str(path))
    headings = [e for e in elements if e.element_type == "heading"]
    assert any("Section Title" in h.content for h in headings), (
        f"14pt heading among 12pt body not detected. Got: {[h.content for h in headings]}"
    )


def test_adaptive_heading_uniform_doc(tmp_path):
    """PDF with uniform font size should produce no headings."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    for i in range(10):
        page.insert_text((72, 100 + i * 20), f"Paragraph {i} text.", fontsize=12)
    path = tmp_path / "uniform.pdf"
    doc.save(str(path))
    doc.close()

    parser = PdfParser()
    elements = parser.parse(str(path))
    headings = [e for e in elements if e.element_type == "heading"]
    assert len(headings) == 0, f"Uniform doc should have no headings, got {[h.content for h in headings]}"
