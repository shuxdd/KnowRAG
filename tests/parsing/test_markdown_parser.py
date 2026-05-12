from backend.services.parsing.markdown_parser import MarkdownParser

SAMPLE_MD = """# Chapter 1

Intro paragraph.

## Section 1.1

More content here.
"""


def test_h1_h2_hierarchy():
    parser = MarkdownParser()
    elements = parser.parse_string(SAMPLE_MD)
    headings = [e for e in elements if e.element_type == "heading"]
    assert len(headings) >= 2
    assert headings[0].content == "Chapter 1"
    assert headings[0].heading_level == 1
    assert headings[1].content == "Section 1.1"
    assert headings[1].heading_level == 2


def test_code_block_preserved():
    parser = MarkdownParser()
    md = """# Title\n\n```python\nx = 1\n```\n\nText"""
    elements = parser.parse_string(md)
    codes = [e for e in elements if e.element_type == "code"]
    assert len(codes) == 1
    assert "x = 1" in codes[0].content


def test_paragraphs():
    parser = MarkdownParser()
    md = "# Title\n\nPara one.\n\nPara two."
    elements = parser.parse_string(md)
    paragraphs = [e for e in elements if e.element_type == "paragraph"]
    assert len(paragraphs) == 2


def test_no_heading():
    parser = MarkdownParser()
    elements = parser.parse_string("Just a paragraph.\n\nAnother one.")
    assert len(elements) > 0
