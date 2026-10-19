"""Self-improving emotion classifier using OpenAI and Arize Prompt Learning.

The script evaluates a prompt on 30 examples from ``dair-ai/emotion``, creates
natural-language feedback, and feeds that feedback to Prompt Learning to refine
the prompt for the next evaluation round.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

try:
    # This is the import path used by the current Arize prompt-learning SDK.
    from optimizer_sdk.prompt_learning_optimizer import PromptLearningOptimizer
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "Arize prompt-learning SDK를 찾을 수 없습니다. README.md의 설치 명령으로 "
        "SDK를 먼저 설치해 주세요."
    ) from exc


LABELS: tuple[str, ...] = (
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
)

INITIAL_PROMPT = """Classify the emotion expressed in the text.
Choose exactly one label from: sadness, joy, love, anger, fear, surprise.
Return only the lowercase label and no explanation.

Text: {text}
Label:"""


def require_openai_api_key() -> str:
    """Return OPENAI_API_KEY or fail with an actionable error message."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.\n"
            "예: export OPENAI_API_KEY='sk-...'"
        )
    return api_key


def load_hf_dataset(sample_count: int = 30) -> pd.DataFrame:
    """Load the first ``sample_count`` examples and map IDs to label names."""
    if sample_count < 1:
        raise ValueError("sample_count는 1 이상이어야 합니다.")

    # Hugging Face split slicing avoids downloading/processing the full train set.
    dataset = load_dataset("dair-ai/emotion", split=f"train[:{sample_count}]")
    frame = dataset.to_pandas()[["text", "label"]].copy()
    frame["expected_label"] = frame["label"].map(dict(enumerate(LABELS)))

    if frame["expected_label"].isna().any():
        unknown = sorted(frame.loc[frame["expected_label"].isna(), "label"].unique())
        raise ValueError(f"알 수 없는 정수 레이블이 발견되었습니다: {unknown}")

    return frame[["text", "expected_label"]].reset_index(drop=True)


def render_prompt(prompt_template: str, text: str) -> str:
    """Insert an input without interpreting any other braces in the prompt."""
    if "{text}" not in prompt_template:
        raise ValueError(
            "프롬프트에 필수 입력 자리표시자 '{text}'가 없습니다. "
            "최적화된 프롬프트를 확인해 주세요."
        )
    return prompt_template.replace("{text}", text)


def normalize_prediction(raw_output: str) -> str:
    """Extract one canonical label from a short or slightly verbose response."""
    cleaned = raw_output.strip().lower()
    if cleaned in LABELS:
        return cleaned

    # Use token boundaries so, for example, a label embedded in another word
    # cannot accidentally count as a correct prediction.
    matches = [
        label
        for label in LABELS
        if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", cleaned)
    ]
    return matches[0] if len(matches) == 1 else "invalid"


def run_inference(
    dataframe: pd.DataFrame,
    prompt_template: str,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """Run deterministic OpenAI inference and add raw/output columns."""
    if not {"text", "expected_label"}.issubset(dataframe.columns):
        raise ValueError("DataFrame에는 text와 expected_label 컬럼이 필요합니다.")

    openai_client = client or OpenAI(api_key=require_openai_api_key())
    evaluated = dataframe.copy()
    raw_outputs: list[str] = []
    outputs: list[str] = []

    for row_number, text in enumerate(evaluated["text"], start=1):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": render_prompt(prompt_template, str(text)),
                    }
                ],
            )
            raw_output = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI 추론 실패 (샘플 {row_number}/{len(evaluated)}): {exc}"
            ) from exc

        raw_outputs.append(raw_output)
        outputs.append(normalize_prediction(raw_output))

    evaluated["raw_output"] = raw_outputs
    evaluated["output"] = outputs
    return evaluated


def generate_feedback(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add correctness and specific natural-language critique for every row."""
    if not {"expected_label", "output"}.issubset(dataframe.columns):
        raise ValueError("DataFrame에는 expected_label과 output 컬럼이 필요합니다.")

    evaluated = dataframe.copy()
    evaluated["is_correct"] = (
        evaluated["expected_label"] == evaluated["output"]
    )

    def build_feedback(row: pd.Series) -> str:
        expected = str(row["expected_label"])
        predicted = str(row["output"])
        if bool(row["is_correct"]):
            return (
                f"Correct. The response selected '{expected}', matching the expected "
                "emotion. Preserve the instruction to return exactly one allowed label."
            )
        if predicted == "invalid":
            return (
                f"Incorrect format or unsupported answer. The expected label was "
                f"'{expected}', but the response did not resolve to exactly one allowed "
                "label. Strengthen the prompt to require one lowercase label only and "
                "use the emotional cues in the text before answering."
            )
        return (
            f"Incorrect. The expected emotion was '{expected}', but the model predicted "
            f"'{predicted}'. Clarify how to distinguish '{expected}' from '{predicted}' "
            "using the text's central emotional state rather than isolated words, and "
            "require exactly one label from the allowed set."
        )

    evaluated["feedback"] = evaluated.apply(build_feedback, axis=1)
    return evaluated


def optimize_prompt(
    current_prompt: str,
    evaluated_data: pd.DataFrame,
    api_key: str,
    meta_model: str = "gpt-4o",
) -> str:
    """Use Arize Prompt Learning to produce the next prompt candidate."""
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

    # Without this placeholder every row would receive the same static prompt.
    if "{text}" not in optimized_prompt:
        raise ValueError(
            "PromptLearningOptimizer가 '{text}' 자리표시자가 없는 프롬프트를 "
            "반환했습니다. 안전을 위해 다음 반복을 중단합니다."
        )
    return optimized_prompt


def save_prompt(prompt: str, output_path: Path) -> None:
    """Persist the final prompt, creating parent directories when necessary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt.rstrip() + "\n", encoding="utf-8")


def run_self_improvement_loop(
    dataframe: pd.DataFrame,
    initial_prompt: str = INITIAL_PROMPT,
    max_iterations: int = 3,
    target_accuracy: float = 0.95,
    output_path: Path = Path("optimized_prompt.txt"),
    inference_model: str = "gpt-4o-mini",
    meta_model: str = "gpt-4o",
    client: Optional[OpenAI] = None,
) -> tuple[str, pd.DataFrame]:
    """Evaluate, critique, and improve a prompt for up to N rounds.

    ``max_iterations`` counts evaluation rounds. Optimization therefore occurs
    between rounds, ensuring that the final returned prompt has been evaluated.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations는 1 이상이어야 합니다.")
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy는 0.0과 1.0 사이여야 합니다.")

    api_key = require_openai_api_key()
    openai_client = client or OpenAI(api_key=api_key)
    current_prompt = initial_prompt
    latest_results = pd.DataFrame()

    print("\n" + "=" * 72)
    print("🚀 Self-Improving Emotion Classifier 시작")
    print(
        f"샘플: {len(dataframe)}개 | 최대 반복: {max_iterations}회 | "
        f"목표 정확도: {target_accuracy:.1%}"
    )
    print("=" * 72)

    for iteration in range(1, max_iterations + 1):
        print(f"\n[Iteration {iteration}/{max_iterations}] 추론 및 평가 중...")
        inferred = run_inference(
            dataframe,
            current_prompt,
            client=openai_client,
            model=inference_model,
        )
        latest_results = generate_feedback(inferred)
        accuracy = float(latest_results["is_correct"].mean())
        correct_count = int(latest_results["is_correct"].sum())
        print(
            f"📊 정확도: {accuracy:.2%} "
            f"({correct_count}/{len(latest_results)})"
        )

        if accuracy >= target_accuracy:
            print(f"🎯 목표 정확도 {target_accuracy:.1%} 달성 — 조기 종료합니다.")
            break

        if iteration == max_iterations:
            print("⏹️ 최대 평가 반복 횟수에 도달했습니다.")
            break

        error_count = len(latest_results) - correct_count
        print(f"📝 오답 {error_count}건의 피드백으로 프롬프트를 최적화합니다...")
        current_prompt = optimize_prompt(
            current_prompt=current_prompt,
            evaluated_data=latest_results,
            api_key=api_key,
            meta_model=meta_model,
        )
        print("\n✨ 개선된 프롬프트")
        print("-" * 72)
        print(current_prompt)
        print("-" * 72)

    save_prompt(current_prompt, output_path)
    print("\n" + "=" * 72)
    print("✅ 최종 프롬프트")
    print("-" * 72)
    print(current_prompt)
    print("-" * 72)
    print(f"💾 저장 위치: {output_path.resolve()}")
    print("=" * 72)
    return current_prompt, latest_results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse optional CLI overrides while keeping useful standalone defaults."""
    parser = argparse.ArgumentParser(
        description="Hugging Face 감정 분류 프롬프트 자가 개선 파이프라인"
    )
    parser.add_argument("--samples", type=int, default=30, help="평가 샘플 수")
    parser.add_argument("--iterations", type=int, default=3, help="최대 평가 반복 수")
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="조기 종료 정확도 (0.0~1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optimized_prompt.txt"),
        help="최종 프롬프트 저장 경로",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the complete pipeline and turn expected failures into clear messages."""
    args = parse_args(argv)
    try:
        # Check credentials before triggering a dataset download.
        require_openai_api_key()
        print("📥 Hugging Face dair-ai/emotion 데이터셋을 불러옵니다...")
        dataframe = load_hf_dataset(sample_count=args.samples)
        print(f"✅ {len(dataframe)}개 샘플 로드 완료")
        run_self_improvement_loop(
            dataframe=dataframe,
            max_iterations=args.iterations,
            target_accuracy=args.target_accuracy,
            output_path=args.output,
        )
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"\n❌ 실행 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
