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


def test_unordered_list_detected():
    parser = MarkdownParser()
    md = "# Title\n\n- Item one\n- Item two\n- Item three"
    elements = parser.parse_string(md)
    lists = [e for e in elements if e.element_type == "list"]
    assert len(lists) == 1
    assert "Item one" in lists[0].content
    assert "Item two" in lists[0].content


def test_ordered_list_detected():
    parser = MarkdownParser()
    md = "# Title\n\n1. First\n2. Second\n3. Third"
    elements = parser.parse_string(md)
    lists = [e for e in elements if e.element_type == "list"]
    assert len(lists) == 1
    assert "First" in lists[0].content


def test_mixed_list_styles():
    parser = MarkdownParser()
    md = "# Title\n\n- Unordered item\n\n1. Ordered item"
    elements = parser.parse_string(md)
    lists = [e for e in elements if e.element_type == "list"]
    assert len(lists) == 2


def test_list_indentation_preserved():
    parser = MarkdownParser()
    md = "# Title\n\n- Top level\n  - Nested level\n    - Deep nested"
    elements = parser.parse_string(md)
    lists = [e for e in elements if e.element_type == "list"]
    assert len(lists) == 1
    assert "  - Nested level" in lists[0].content
    assert "    - Deep nested" in lists[0].content
