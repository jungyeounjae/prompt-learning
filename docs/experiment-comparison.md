# 반복 실행 비교

감정 분류 실험은 라벨당 100개 표본을 사용하고, seed만 다른 세 실행으로 결과의 변동을 확인한다.

```bash
.venv/bin/prompt-learning run --config configs/emotion.yaml
.venv/bin/prompt-learning run --config configs/emotion-seed-123.yaml
.venv/bin/prompt-learning run --config configs/emotion-seed-2024.yaml
```

각 실행이 끝나면 출력된 `runs/emotion/<run-id>` 경로를 비교 명령에 전달한다.

```bash
.venv/bin/prompt-learning compare \
  runs/emotion/<run-id-1> runs/emotion/<run-id-2> runs/emotion/<run-id-3> \
  --output artifacts/emotion-comparison.csv
```

명령은 실제 산출물의 seed, 표본 설정, Validation/Test accuracy와 macro-F1만 모아
`emotion-comparison.csv`, Markdown 표, 그리고 `emotion-comparison-summary.csv`를 생성한다.
요약 CSV에는 seed별 평균·최솟값·최댓값·모집단 표준편차가 포함된다. 실행하지 않은 값은
추정하거나 채우지 않는다. 세 seed의 평균과 범위는 표를 바탕으로 직접 계산해,
단일 실행의 우연한 결과와 구분한다.
