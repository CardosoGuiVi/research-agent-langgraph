"""Live mini-eval: `make eval` (calls paid APIs; roughly $0.3 per question).

For each question in evals/dataset.yaml: run the full graph, apply deterministic checks
(schema, citations resolve to retrieved sources, distinct source domains) and an LLM-as-judge
score (fast model, rubric in src/research_agent/prompts/judge.md). Full results are written to
evals/results/ (gitignored: they contain raw model output); a markdown summary is printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.checks import JudgeScore, run_checks
from research_agent.config import LLMProviderName, Settings, get_settings
from research_agent.container import Container, build_container
from research_agent.llm.base import Tier
from research_agent.logging import configure_logging
from research_agent.prompts import load_prompt

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


async def judge(
    container: Container, question: str, expected: list[str], markdown: str
) -> JudgeScore:
    prompt = load_prompt("judge")
    score, _ = await container.service.deps.llm.structured(
        JudgeScore,
        system=prompt.system,
        user=prompt.render_user(question=question, expected=", ".join(expected), report=markdown),
        tier=Tier.FAST,
        max_tokens=1_000,
    )
    return score


async def evaluate(container: Container, item: dict[str, Any], stamp: str) -> dict[str, Any]:
    thread_id = f"eval-{item['id']}-{stamp}"
    start = time.perf_counter()
    try:
        result = await container.service.run(item["question"], thread_id)
        view = await container.service.get_thread(thread_id)
        sources = view.sources if view else []
        report_json = result.report.model_dump(mode="json")
        checks = run_checks(report_json, sources, item.get("min_sources", 3))
        score = await judge(
            container, item["question"], item.get("expected", []), result.report.markdown
        )
        return {
            "id": item["id"],
            "ok": True,
            "checks": asdict(checks),
            "judge": score.model_dump(),
            "usage": result.usage,
            "report": report_json,
            "latency_s": round(time.perf_counter() - start, 1),
        }
    except Exception as exc:  # one failing question must not abort the eval
        return {
            "id": item["id"],
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "latency_s": round(time.perf_counter() - start, 1),
        }


def describe_models(settings: Settings) -> str:
    """The models actually in use, for the active LLM_PROVIDER."""
    if settings.llm_provider is LLMProviderName.OPENROUTER:
        smart = settings.openrouter_model_smart or settings.openrouter_model
        return f"provider=openrouter, fast={settings.openrouter_model}, smart={smart}"
    return (
        f"provider=anthropic, fast={settings.llm_model_fast}, smart={settings.llm_model_smart} "
        f"(effort {settings.llm_smart_effort})"
    )


def summary_table(rows: list[dict[str, Any]], model_info: str) -> str:
    def yes(flag: bool) -> str:
        return "yes" if flag else "**no**"

    lines = [
        f"_Models: {model_info}_",
        "",
        "| Question | Schema | Citations resolve | Distinct domains | Relevance | Completeness"
        " | Cost (USD) | Latency (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ok = [r for r in rows if r["ok"]]
    for r in rows:
        if not r["ok"]:
            lines.append(f"| {r['id']} | error: {r['error'][:60]} | | | | | | |")
            continue
        c, j, u = r["checks"], r["judge"], r["usage"]
        cost = u.get("estimated_cost_usd")
        cost_str = f"{cost:.3f}" if cost is not None else "n/a"
        lines.append(
            f"| {r['id']} | {yes(c['schema_valid'])} | {yes(c['citations_resolve'])} "
            f"| {c['distinct_domains']}{'' if c['min_sources_ok'] else ' (below min)'} "
            f"| {j['relevance']}/5 | {j['completeness']}/5 "
            f"| {cost_str} | {r['latency_s']:.0f} |"
        )
    if ok:
        n = len(ok)

        def avg(key: str) -> float:
            return sum(r["judge"][key] for r in ok) / n

        passed = sum(
            r["checks"]["schema_valid"]
            and r["checks"]["citations_resolve"]
            and r["checks"]["min_sources_ok"]
            for r in ok
        )
        total_cost = sum(r["usage"].get("estimated_cost_usd") or 0 for r in ok)
        lines.append(
            f"| **Total ({n}/{len(rows)} ran)** | | **{passed}/{n} pass all checks** | "
            f"| **{avg('relevance'):.1f}** | **{avg('completeness'):.1f}** "
            f"| **{total_cost:.2f}** | {sum(r['latency_s'] for r in ok) / n:.0f} avg |"
        )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="question ids to run")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", json=True)
    container = build_container(settings)
    dataset = yaml.safe_load((ROOT / "dataset.yaml").read_text(encoding="utf-8"))
    items = [q for q in dataset["questions"] if not args.only or q["id"] in args.only]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            print(f"running {item['id']} ...", flush=True)
            return await evaluate(container, item, stamp)

    try:
        rows = await asyncio.gather(*(bounded(i) for i in items))
    finally:
        await container.aclose()

    model_info = (
        f"{describe_models(settings)}, search={settings.search_provider.value}, "
        f"max_iterations={settings.max_iterations}, prompts="
        + ", ".join(
            f"{n}@v{load_prompt(n).version}"
            for n in ("planner", "research", "analysis", "answer", "judge")
        )
    )
    table = summary_table(rows, model_info)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"run-{stamp}.json").write_text(
        json.dumps({"stamp": stamp, "config": model_info, "rows": rows}, indent=2), encoding="utf-8"
    )
    (RESULTS / "latest.md").write_text(f"Run {stamp}\n\n{table}\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    asyncio.run(main())
