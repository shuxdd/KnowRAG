from backend.services.parsing.txt_parser import TxtParser


def test_paragraph_split():
    parser = TxtParser()
    text = "Para one.\n\nPara two.\n\nPara three."
    elements = parser.parse_string(text)
    assert len(elements) == 3
    assert all(e.element_type == "paragraph" for e in elements)


def test_no_heading_elements():
    parser = TxtParser()
    elements = parser.parse_string("Just text.")
    headings = [e for e in elements if e.element_type == "heading"]
    assert len(headings) == 0
