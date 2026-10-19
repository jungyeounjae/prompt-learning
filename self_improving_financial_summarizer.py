"""Self-improving financial-news summarizer using OpenAI and Arize.

The pipeline generates summaries, asks an evaluator LLM for structured natural-
language critique, and uses Arize Prompt Learning to improve the summarization
prompt between evaluation rounds.  Every round is saved for later analysis and
for reproducing charts/tables in a technical blog post.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

try:
    from optimizer_sdk.prompt_learning_optimizer import PromptLearningOptimizer
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "Arize prompt-learning SDK를 불러올 수 없습니다. "
        "`.venv/bin/python -m pip install -r requirements.txt`를 실행해 주세요."
    ) from exc


# The two IDs originally proposed in the specification are not currently
# available as public Hugging Face datasets. This public financial-news dataset
# provides Content (article) and Summary (reference) columns.
DEFAULT_DATASET = "kdave/Indian_Financial_News"

INITIAL_PROMPT = """Summarize the following financial news article:

Article: {article}

Summary:"""

EVALUATOR_PROMPT = """You are a strict evaluator of financial-news summaries.

Evaluate the candidate summary using the source article and the reference
summary. The reference is a useful comparison, but the source article is the
ultimate factual authority.

Source article:
---
{article}
---

Reference summary:
---
{reference_summary}
---

Candidate summary:
---
{output}
---

Score each dimension with an integer from 1 (poor) to 5 (excellent):
1. financial_figures_score: Includes material revenue, profit, growth rate,
   price, guidance, or other important figures when they exist in the article.
2. conciseness_score: Focuses on the essential information without repetition,
   filler, or unnecessary commentary.
3. faithfulness_score: Contains no unsupported claims and accurately represents
   the source article.

Then write actionable natural-language feedback. Explicitly identify omissions,
unnecessary details, and hallucinations. Give concrete prompt rules that would
prevent the observed problems. If the summary is strong, state what should be
preserved. Do not give financial advice.

Return only a JSON object with exactly these fields:
{
  "financial_figures_score": 1,
  "conciseness_score": 1,
  "faithfulness_score": 1,
  "feedback": "Specific evaluation and improvement rules"
}
"""


def require_openai_api_key() -> str:
    """Load .env and return OPENAI_API_KEY with an actionable failure message."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "sk-your-api-key-here":
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. 프로젝트 루트의 .env에 "
            "`OPENAI_API_KEY=sk-...` 형식으로 실제 키를 입력해 주세요."
        )
    return api_key


def _find_column(columns: Sequence[str], aliases: Sequence[str]) -> str:
    """Find a dataset column case-insensitively from known aliases."""
    lookup = {column.casefold(): column for column in columns}
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    raise ValueError(
        f"필요한 컬럼을 찾지 못했습니다. 후보={list(aliases)}, "
        f"실제 컬럼={list(columns)}"
    )


def load_financial_dataset(
    dataset_name: str = DEFAULT_DATASET,
    sample_count: int = 20,
    split: str = "train",
) -> pd.DataFrame:
    """Load and normalize a Hugging Face financial summarization dataset."""
    if sample_count < 1:
        raise ValueError("sample_count는 1 이상이어야 합니다.")

    try:
        dataset = load_dataset(dataset_name, split=f"{split}[:{sample_count}]")
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face 데이터셋 '{dataset_name}' 로드에 실패했습니다: {exc}"
        ) from exc
    frame = dataset.to_pandas()
    article_column = _find_column(
        list(frame.columns),
        ("article", "content", "text", "document", "source"),
    )
    summary_column = _find_column(
        list(frame.columns),
        ("reference_summary", "summary", "highlights", "target", "abstract"),
    )

    normalized = frame[[article_column, summary_column]].rename(
        columns={
            article_column: "article",
            summary_column: "reference_summary",
        }
    )
    normalized = normalized.dropna().copy()
    normalized["article"] = normalized["article"].astype(str).str.strip()
    normalized["reference_summary"] = (
        normalized["reference_summary"].astype(str).str.strip()
    )
    normalized = normalized[
        normalized["article"].ne("") & normalized["reference_summary"].ne("")
    ]

    if normalized.empty:
        raise ValueError("유효한 기사와 참고 요약이 포함된 행이 없습니다.")
    return normalized.reset_index(drop=True)


def render_article_prompt(prompt_template: str, article: str) -> str:
    """Safely insert an article while preserving unrelated literal braces."""
    if "{article}" not in prompt_template:
        raise ValueError("프롬프트에 필수 자리표시자 '{article}'가 없습니다.")
    return prompt_template.replace("{article}", article)


def run_summarization(
    dataframe: pd.DataFrame,
    prompt_template: str,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """Generate one deterministic summary per article in an output column."""
    required = {"article", "reference_summary"}
    if not required.issubset(dataframe.columns):
        raise ValueError(f"DataFrame에는 {sorted(required)} 컬럼이 필요합니다.")

    openai_client = client or OpenAI(api_key=require_openai_api_key())
    results = dataframe.copy()
    outputs: list[str] = []

    for position, article in enumerate(results["article"], start=1):
        print(f"  요약 생성: {position}/{len(results)}", end="\r", flush=True)
        try:
            response = openai_client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": render_article_prompt(
                            prompt_template, str(article)
                        ),
                    }
                ],
            )
            output = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise RuntimeError(
                f"요약 생성 실패 ({position}/{len(results)}): {exc}"
            ) from exc
        if not output:
            raise RuntimeError(f"요약 생성 결과가 비어 있습니다 (샘플 {position}).")
        outputs.append(output)

    print(" " * 60, end="\r")
    results["output"] = outputs
    return results


def _parse_evaluation_json(raw_response: str, row_number: int) -> dict[str, object]:
    """Validate the evaluator's JSON and normalize scores."""
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"평가자 응답이 JSON이 아닙니다 (샘플 {row_number}): "
            f"{raw_response[:200]}"
        ) from exc

    score_fields = (
        "financial_figures_score",
        "conciseness_score",
        "faithfulness_score",
    )
    normalized: dict[str, object] = {}
    for field in score_fields:
        try:
            value = int(payload[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"평가자 응답의 {field} 값이 유효하지 않습니다 "
                f"(샘플 {row_number})."
            ) from exc
        if not 1 <= value <= 5:
            raise RuntimeError(
                f"{field}는 1~5 범위여야 합니다 (샘플 {row_number}, 값={value})."
            )
        normalized[field] = value

    feedback = str(payload.get("feedback", "")).strip()
    if not feedback:
        raise RuntimeError(f"평가자 feedback이 비어 있습니다 (샘플 {row_number}).")
    normalized["feedback"] = feedback
    return normalized


def generate_critique(
    dataframe: pd.DataFrame,
    client: Optional[OpenAI] = None,
    evaluator_model: str = "gpt-4o",
) -> pd.DataFrame:
    """Evaluate summaries with an LLM and add scores plus natural critique."""
    required = {"article", "reference_summary", "output"}
    if not required.issubset(dataframe.columns):
        raise ValueError(f"DataFrame에는 {sorted(required)} 컬럼이 필요합니다.")

    openai_client = client or OpenAI(api_key=require_openai_api_key())
    evaluations: list[dict[str, object]] = []

    for row_number, row in enumerate(dataframe.itertuples(index=False), start=1):
        print(f"  평가 생성: {row_number}/{len(dataframe)}", end="\r", flush=True)
        evaluator_input = (
            EVALUATOR_PROMPT.replace("{article}", str(row.article))
            .replace("{reference_summary}", str(row.reference_summary))
            .replace("{output}", str(row.output))
        )
        try:
            response = openai_client.chat.completions.create(
                model=evaluator_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": evaluator_input}],
            )
            raw_response = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise RuntimeError(
                f"평가자 호출 실패 ({row_number}/{len(dataframe)}): {exc}"
            ) from exc
        evaluations.append(_parse_evaluation_json(raw_response, row_number))

    print(" " * 60, end="\r")
    evaluated = dataframe.copy()
    evaluation_frame = pd.DataFrame(evaluations, index=evaluated.index)
    for column in evaluation_frame.columns:
        evaluated[column] = evaluation_frame[column]

    # 0~100 scale derived from three independently scored dimensions.
    score_columns = [
        "financial_figures_score",
        "conciseness_score",
        "faithfulness_score",
    ]
    evaluated["overall_score"] = evaluated[score_columns].mean(axis=1) * 20.0
    return evaluated


def optimize_prompt(
    current_prompt: str,
    evaluated_data: pd.DataFrame,
    api_key: str,
    meta_model: str = "gpt-4o",
) -> str:
    """Create a revised summarization prompt from outputs and critiques."""
    optimizer = PromptLearningOptimizer(
        prompt=current_prompt,
        model_choice=meta_model,
        openai_api_key=api_key,
    )
    optimized = optimizer.optimize(
        dataset=evaluated_data,
        output_column="output",
        feedback_columns=["feedback"],
    )
    optimized_prompt = str(optimized).strip()
    if "{article}" not in optimized_prompt:
        raise ValueError(
            "최적화된 프롬프트에 '{article}' 자리표시자가 없습니다. "
            "잘못된 프롬프트 적용을 막기 위해 중단합니다."
        )
    return optimized_prompt


def _save_iteration_artifacts(
    output_dir: Path,
    iteration: int,
    prompt: str,
    evaluated_data: pd.DataFrame,
) -> None:
    """Save reproducible per-round inputs, outputs, critiques, and scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"iteration_{iteration}_prompt.txt").write_text(
        prompt.rstrip() + "\n", encoding="utf-8"
    )
    evaluated_data.to_csv(
        output_dir / f"iteration_{iteration}_results.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _archive_arize_metaprompt(output_dir: Path, optimization_number: int) -> None:
    """Copy the SDK's latest metaprompt before the next optimization overwrites it."""
    generated_path = Path("metaprompt.txt")
    if generated_path.exists():
        shutil.copy2(
            generated_path,
            output_dir / f"optimization_{optimization_number}_metaprompt.txt",
        )


def run_self_improving_loop(
    dataframe: pd.DataFrame,
    initial_prompt: str = INITIAL_PROMPT,
    iterations: int = 3,
    output_dir: Path = Path("financial_results"),
    final_prompt_path: Path = Path("financial_optimized_prompt.txt"),
    summarizer_model: str = "gpt-4o-mini",
    evaluator_model: str = "gpt-4o",
    meta_model: str = "gpt-4o",
    client: Optional[OpenAI] = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Run N evaluated rounds and return the best prompt, results, and history."""
    if iterations < 1:
        raise ValueError("iterations는 1 이상이어야 합니다.")

    api_key = require_openai_api_key()
    openai_client = client or OpenAI(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_prompt = initial_prompt
    best_prompt = initial_prompt
    best_score = float("-inf")
    best_results = pd.DataFrame()
    history: list[dict[str, object]] = []

    print("\n" + "=" * 80)
    print("금융 뉴스 요약 프롬프트 Self-Improvement 시작")
    print(
        f"샘플={len(dataframe)} | 평가 회차={iterations} | "
        f"요약={summarizer_model} | 평가/최적화={evaluator_model}/{meta_model}"
    )
    print("=" * 80)

    for iteration in range(1, iterations + 1):
        print("\n" + "=" * 80)
        print(f"Iteration {iteration}/{iterations}: 요약 생성")
        print("=" * 80)
        summarized = run_summarization(
            dataframe,
            current_prompt,
            client=openai_client,
            model=summarizer_model,
        )

        print("\n" + "=" * 80)
        print(f"Iteration {iteration}/{iterations}: Evaluator 평가")
        print("=" * 80)
        evaluated = generate_critique(
            summarized,
            client=openai_client,
            evaluator_model=evaluator_model,
        )
        mean_score = float(evaluated["overall_score"].mean())
        dimension_means = {
            "financial_figures": float(evaluated["financial_figures_score"].mean()),
            "conciseness": float(evaluated["conciseness_score"].mean()),
            "faithfulness": float(evaluated["faithfulness_score"].mean()),
        }
        print(f"평균 종합 점수: {mean_score:.2f}/100")
        print(
            "세부 평균(5점): "
            f"재무수치={dimension_means['financial_figures']:.2f}, "
            f"간결성={dimension_means['conciseness']:.2f}, "
            f"충실성={dimension_means['faithfulness']:.2f}"
        )
        print("\n평가자 피드백 샘플:")
        print("-" * 80)
        print(str(evaluated.iloc[0]["feedback"]))
        print("-" * 80)

        _save_iteration_artifacts(output_dir, iteration, current_prompt, evaluated)
        history.append(
            {
                "iteration": iteration,
                "overall_score": round(mean_score, 4),
                **{key: round(value, 4) for key, value in dimension_means.items()},
                "prompt_file": f"iteration_{iteration}_prompt.txt",
                "results_file": f"iteration_{iteration}_results.csv",
            }
        )

        # Keep the best evaluated prompt instead of assuming later is always better.
        if mean_score > best_score:
            best_score = mean_score
            best_prompt = current_prompt
            best_results = evaluated.copy()

        # Do not create an unevaluated prompt after the final round.
        if iteration == iterations:
            break

        print("\n" + "=" * 80)
        print(f"Iteration {iteration}: Arize Prompt Learning 최적화")
        print("=" * 80)
        current_prompt = optimize_prompt(
            current_prompt,
            evaluated,
            api_key=api_key,
            meta_model=meta_model,
        )
        _archive_arize_metaprompt(output_dir, iteration)
        print("새로 업데이트된 프롬프트:")
        print("-" * 80)
        print(current_prompt)
        print("-" * 80)

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        output_dir / "iteration_history.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "iteration_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    final_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    final_prompt_path.write_text(best_prompt.rstrip() + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"최고 평가 프롬프트 (평균 점수 {best_score:.2f}/100)")
    print("=" * 80)
    print(best_prompt)
    print("=" * 80)
    print(f"최종 프롬프트 저장: {final_prompt_path.resolve()}")
    print(f"회차별 결과 저장: {output_dir.resolve()}")
    return best_prompt, best_results, history_frame


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Expose cost- and dataset-related settings as CLI options."""
    parser = argparse.ArgumentParser(
        description="금융 뉴스 요약 프롬프트 자가 개선 파이프라인"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--summarizer-model", default="gpt-4o-mini")
    parser.add_argument("--evaluator-model", default="gpt-4o")
    parser.add_argument("--meta-model", default="gpt-4o")
    parser.add_argument("--output-dir", type=Path, default=Path("financial_results"))
    parser.add_argument(
        "--final-prompt",
        type=Path,
        default=Path("financial_optimized_prompt.txt"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the complete standalone pipeline with readable error reporting."""
    args = parse_args(argv)
    try:
        require_openai_api_key()
        print(f"Hugging Face 데이터셋 로드: {args.dataset}")
        dataframe = load_financial_dataset(
            dataset_name=args.dataset,
            sample_count=args.samples,
            split=args.split,
        )
        print(f"유효한 금융 기사 {len(dataframe)}개 로드 완료")
        run_self_improving_loop(
            dataframe=dataframe,
            iterations=args.iterations,
            output_dir=args.output_dir,
            final_prompt_path=args.final_prompt,
            summarizer_model=args.summarizer_model,
            evaluator_model=args.evaluator_model,
            meta_model=args.meta_model,
        )
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"\n실행 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
