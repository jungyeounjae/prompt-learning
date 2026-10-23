[日本語](README.md) | [한국어](README.ko.md)

# 評価に基づくプロンプト最適化実験

プロンプトの変更は平均点を上げても、一部の入力や評価基準を悪化させることが
あります。このツールは感情分類と金融ニュース要約について、Trainで候補を
生成し、独立したValidationの回帰防止条件で採否を決め、最終プロンプトだけを
Testで1回評価します。ローカルCLIと成果物で実験を再現・検討できます。

現時点で、この共通パイプラインによる実モデル実行は検証できていません。
性能向上、実測コスト・遅延、Phoenix画面の証拠はありません。
[実験の状態と旧成果物](artifacts/experiment-summary.md)を参照してください。
自動テストは外部サービスを置き換えた検証であり、モデル品質の測定ではありません。

## セットアップとオフライン確認

リポジトリのルートからPython 3.12以上で実行します。依存関係の正本は
[pyproject.toml](pyproject.toml)です。[requirements.txt](requirements.txt)は
同じプロジェクトをインストールする互換用ファイルです。初回インストールには
ネットワークと上流Gitリポジトリへのアクセスが必要です。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' build
# 互換用の基本インストール: python -m pip install -r requirements.txt
```

以下のヘルプはAPIキーもネットワークも必要としません。

```bash offline
python -m prompt_learning.cli --help
python -m prompt_learning.cli run --help
```

インストール後は `prompt-learning --help` も利用できます。

## モデル実験の実行

`OPENAI_API_KEY`を環境変数かルートのローカル`.env`に設定してください。
キーを履歴に残さない入力例です。以下の実験は有料APIとデータセットの
ダウンロードを使用します。

```bash
read -r -s OPENAI_API_KEY
export OPENAI_API_KEY
export PHOENIX_ENABLED=false
python -m prompt_learning.cli run --config configs/emotion.yaml
python -m prompt_learning.cli run --config configs/financial.yaml
```

基本設定はseed 42、60/20/20分割、最大3候補、感情は各ラベル20件
（計120件）、金融は重複除去後60件です。推論は`gpt-4o-mini`、
金融評価と候補生成は`gpt-4o`です。感情評価にLLMは使用しません。

小規模確認では、まずHugging Faceの現在のcommitを解決して固定します。
同じ実験を再現するときは、保存済み設定のcommitを再利用してください。

```bash
export EMOTION_DATASET_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("dair-ai/emotion").sha)')"
export FINANCIAL_DATASET_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("kdave/Indian_Financial_News").sha)')"
```

次の設定生成自体はオフラインです。上で解決したrevisionが必要です。
元の設定を保持し、感情は各ラベル5件（計30件）、金融は5件に減らします。
全モデルを`gpt-4o-mini`、候補は1回とし、splitと承認基準を保持します。

```bash offline
python - <<'PY'
import os
from pathlib import Path
import yaml

destination = Path('runs/configs')
destination.mkdir(parents=True, exist_ok=True)
for task, count_key in (('emotion', 'samples_per_label'), ('financial', 'sample_count')):
    config = yaml.safe_load(Path(f'configs/{task}.yaml').read_text())
    config['dataset']['revision'] = os.environ[f'{task.upper()}_DATASET_REVISION']
    config['dataset'][count_key] = 5
    config['models'].update(inference='gpt-4o-mini', evaluator='gpt-4o-mini', meta='gpt-4o-mini')
    config['max_iterations'] = 1
    config['phoenix']['enabled'] = False
    config['cache_path'] = f'.cache/{task}-smoke.sqlite3'
    (destination / f'{task}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

```bash
export PHOENIX_ENABLED=false
python -m prompt_learning.cli run --config runs/configs/emotion.yaml
python -m prompt_learning.cli run --config runs/configs/financial.yaml
```

これは実行確認用の最小規模です。金融の分割は3/1/1件となり、性能の一般化を
推定するには小さすぎます。デフォルトの`revision: main`、モデル名のalias、
未固定の上流依存関係は変化します。厳密な追試にはrevision、モデルsnapshot、
依存関係一覧、設定ファイルを保存してください。

## 評価と候補の選択

Trainは候補生成だけ、Validationは選択だけ、Testは最終評価だけに使用します。
Trainの点数は最終性能ではありません。Testを見て候補や閾値を調整すると
独立した評価ではなくなり、新しい未使用データが必要になります。

| タスク | Validationの承認条件 |
|---|---|
| 感情分類 | Macro F1非低下、Accuracy低下は1ポイント以内、invalid率非増加、各クラスRecall低下は5ポイント未満 |
| 金融要約 | 総合点・平均Faithfulness非低下、深刻な幻覚数非増加、20点以上の低下例が10%以内 |

同点は短いプロンプトを優先し、長さも同じなら現在の候補を保持します。
金融は財務数値・簡潔性・忠実性を1〜5点で評価し、財務数値が存在しない記事は
その基準を`not_applicable`として分母から除外します。空出力、40〜800字の
範囲、JSON形式、原文にない通貨・百分率も検査します。
拒否した候補も保存するため、平均点だけでは見えない回帰を確認できます。
閾値は実装済みの承認ポリシーに固定され、任意の変更値はCLIで拒否されます。

## Phoenixとローカル成果物

Phoenixは任意です。`PHOENIX_ENABLED=false`では観測接続を行いません。
有効時の接続失敗は警告となり、ローカル実験は継続します。

```bash
python -m pip install -e '.[phoenix]'
docker compose config
docker compose up -d
export PHOENIX_ENABLED=true
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
python -m prompt_learning.cli run --config runs/configs/emotion.yaml
docker compose stop
```

UIは`http://localhost:6006`です。Composeの既定imageは`latest`なので、
追試では`PHOENIX_IMAGE`に確認済みtag/digestを指定してください。
データはDocker volumeに残ります。Phoenixにはhash/IDと集計情報を送り、
原文や参照要約を送らない設計です。実サーバー互換性と画面は未検証です。

既定の保存先は`runs/<task>/<run-id>/`です。manifestにはUTC時刻、設定、
revision、モデル、seed、入力hashを記録します。プロンプト、各候補の
Validation指標、最終Test指標、拒否理由、回帰一覧、レポートを保存し、
感情分類には混同行列画像も生成します。`runs/`と`.cache/`はGit対象外です。
[アーキテクチャと成果物の仕様](docs/architecture.md)に詳細があります。

## 限界と検証

LLM評価のバイアス、少数例の分散、Validationへの繰り返し適合が残ります。
金融評価は人手監査と照合してください。Testは選択されたプロンプトだけを
評価するため、同じTest上の初期モデルとの差はこの実行だけでは測定しません。
価格設定は既定で空で、コストは不明です。現在のCLIのローカル成果物には
完全なトークン・費用・呼び出し遅延の台帳がなく、meta modelも共通gatewayの
外で呼び出されます。実行時間だけをAPI遅延や実費として報告しないでください。
公開前にプロンプト内の例やスクリーンショットも点検してください。

```bash
python -m pytest tests/test_documentation.py -v
python -m pytest -v
python -m build
docker compose config
git diff --check
```

旧`self_improving_*.py`は非推奨の互換ラッパーです。`--samples 5`の旧感情例や
単一出力ファイル指定は使えません。新しいCLIと設定に移行してください。
実装済み機能と未検証事項は[実験状態](artifacts/experiment-summary.md)を参照してください。
