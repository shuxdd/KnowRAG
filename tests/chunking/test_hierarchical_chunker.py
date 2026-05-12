import uuid
from backend.services.chunking.hierarchical_chunker import HierarchicalChunker
from backend.services.parsing.base import StructuredElement


def _make_heading(text: str, level: int) -> StructuredElement:
    return StructuredElement(content=text, element_type="heading", heading_level=level)


def _make_para(text: str) -> StructuredElement:
    return StructuredElement(content=text, element_type="paragraph")


def _make_code(text: str) -> StructuredElement:
    return StructuredElement(content=text, element_type="code")


def test_h2_is_parent_boundary():
    elements = [
        _make_heading("H1 Title", 1),
        _make_para("Lead."),
        _make_heading("Section 1.1", 2),
        _make_para("Content A."),
        _make_heading("Section 1.2", 2),
        _make_para("Content B."),
    ]
    chunker = HierarchicalChunker()
    parents, leaves = chunker.chunk(elements, "test.md")
    # H1 + two h2 boundaries produce 3 parents: Lead | Section 1.1 | Section 1.2
    assert len(parents) == 3, f"expected 3 parents, got {len(parents)}"
    assert "Content A" in parents[1].content
    assert "Content B" in parents[2].content


def test_h3_stays_in_parent():
    elements = [
        _make_heading("Chapter", 2),
        _make_para("Intro."),
        _make_heading("Subsection", 3),
        _make_para("Detail."),
    ]
    chunker = HierarchicalChunker()
    parents, leaves = chunker.chunk(elements, "test.md")
    assert len(parents) == 1
    assert "Subsection" in parents[0].content


def test_implicit_parent_no_headings():
    elements = [_make_para("Just text.")]
    chunker = HierarchicalChunker()
    parents, leaves = chunker.chunk(elements, "plain.txt")
    assert len(parents) == 1
    assert parents[0].heading_path == ["plain.txt"]


def test_table_code_preserve():
    elements = [
        _make_heading("Section", 2),
        _make_para("Text."),
        _make_code("x = 1"),
        _make_para("More."),
    ]
    chunker = HierarchicalChunker()
    parents, leaves = chunker.chunk(elements, "test.md")
    preserve_leaves = [l for l in leaves if l.preserve]
    assert len(preserve_leaves) >= 1
    assert "x = 1" in preserve_leaves[0].content


def test_parent_leaf_link():
    elements = [
        _make_heading("A", 2),
        _make_para("Hello world." * 50),
    ]
    chunker = HierarchicalChunker()
    parents, leaves = chunker.chunk(elements, "big.md")
    assert len(parents) == 1
    for leaf in leaves:
        assert leaf.parent_id == parents[0].id
    assert len(leaves) >= 1


def test_parent_oversize_downgrade():
    elements = [
        _make_heading("Big Section", 2),
        _make_para("A" * 1600),
    ]
    chunker = HierarchicalChunker()
    parents, leaves = chunker.chunk(elements, "big.md")
    assert len(parents) >= 2, "oversize parent should be split"


def test_parent_oversize_with_h3():
    elements = [
        _make_heading("Big Section", 2),
        _make_heading("Sub A", 3),
        _make_para("A" * 1600),
        _make_heading("Sub B", 3),
        _make_para("B" * 1600),
    ]
    chunker = HierarchicalChunker()
    parents, leaves = chunker.chunk(elements, "big.md")
    assert len(parents) >= 2
