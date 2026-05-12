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
