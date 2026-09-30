import pytest

from research_agent.prompts import PROMPT_NAMES, load_prompt


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_every_prompt_has_a_version_and_body(name: str) -> None:
    prompt = load_prompt(name)
    assert prompt.version >= 1
    assert prompt.system.strip()


def test_render_substitutes_variables() -> None:
    prompt = load_prompt("planner")
    text = prompt.render_user(question="What is X?", prior_context="(none)", min_n=3, max_n=6)
    assert "What is X?" in text
    assert "$" not in text.replace("$$", "")


def test_render_rejects_missing_variables() -> None:
    with pytest.raises(KeyError):
        load_prompt("planner").render_user(question="q")
