from pathlib import Path

from research_agent.graph.export import main, mermaid


def test_mermaid_contains_all_nodes_and_edges() -> None:
    text = mermaid()
    for node in ("planner", "research", "analysis", "answer"):
        assert node in text
    assert "__start__" in text
    assert "__end__" in text


def test_main_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "graph.mmd"
    main(["export", str(out)])
    assert out.read_text().startswith("---") or "graph" in out.read_text()
