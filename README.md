[日本語](README.md) | [한국어](README.ko.md)

# Arize Prompt Learningによる自己改善型LLMプロンプト

Hugging Faceのデータセット、OpenAIモデル、Arize Prompt Learningを使い、
自然言語フィードバックをもとにLLMプロンプトを反復的に改善する実験リポジトリです。

次の2つのタスクを検証します。

1. **dair-ai/emotion**を使用した感情分類
2. **kdave/Indian_Financial_News**を使用した金融ニュース要約

共通のパイプラインは以下のとおりです。

~~~text
現在のプロンプトでAgentを実行
          ↓
結果を評価し、自然言語フィードバックを生成
          ↓
Arize PromptLearningOptimizer.optimize()
          ↓
GPT-4oが改善プロンプトを生成
          ↓
次のIterationで再評価
~~~

## セットアップ

現在、Arize Prompt Learningの公式リポジトリ版パッケージはPython 3.12以上を
必要とします。本プロジェクトのスクリプト自体はPython 3.10の構文と互換性が
ありますが、最新SDKと一緒にインストールする場合はPython 3.12以上の
仮想環境を使用してください。

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
~~~

## 実行方法

デフォルト設定では、先頭30サンプル、最大3回の評価、目標精度95%を使用します。

~~~bash
python self_improving_emotion_classifier.py
~~~

APIコストを抑えて短時間で確認する場合は、次のように実行できます。

~~~bash
python self_improving_emotion_classifier.py --samples 5 --iterations 1
~~~

最終プロンプトはデフォルトで **optimized_prompt.txt** に保存されます。
**--output**、**--iterations**、**--target-accuracy** オプションで設定を変更できます。

## 実験1：感情分類の結果

簡易検証として **dair-ai/emotion** の先頭5サンプルを実行したところ、
初期プロンプトは2件に正解し、精度は40%でした。

| 入力の主な表現 | 正解 | 初期予測 | 結果 |
|---|---|---|---|
| **humiliated** | sadness | sadness | 正解 |
| **hopeless**から**hopeful**への変化 | sadness | joy | 不正解 |
| **greedy**、**wrong** | anger | sadness | 不正解 |
| 暖炉を回想する文 | love | sadness | 不正解 |
| **grouchy** | anger | anger | 正解 |

初期モデルは文全体の感情よりも、**hopeful**、**wrong**、**nostalgic**などの
一部の単語に強く反応し、**joy**、**anger**、**love**を**sadness**と
混同していました。

Arize Prompt Learningはフィードバックをもとに、次の指示を追加しました。

~~~text
Consider the overall emotional tone of the text rather than focusing on
isolated words.
~~~

さらに、3件の誤答を含む実際のサンプルをFew-shot例として追加しました。
生成されたプロンプト全体は
[artifacts/emotion/optimized_prompt.txt](artifacts/emotion/optimized_prompt.txt)
で確認できます。

この実行では、改善プロンプトを独立したTestデータで再検証していません。
そのため、40%から汎化性能が向上したとは結論づけていません。今後は
最適化用Trainデータと評価用Testデータを分離する必要があります。

## 金融ニュース要約パイプライン

**self_improving_financial_summarizer.py** は金融記事をGPT-4o-miniで要約し、
GPT-4o Evaluatorの自然言語による批評を使って、Arize Prompt Learningで
プロンプトを改善します。デフォルト実行では20記事を3回評価するため、
まずは次の低コストな確認コマンドを推奨します。

~~~bash
python self_improving_financial_summarizer.py \
  --samples 2 \
  --iterations 2 \
  --evaluator-model gpt-4o-mini
~~~

デフォルト設定ですべて実行する場合は、次のコマンドを使用します。

~~~bash
python self_improving_financial_summarizer.py
~~~

最終プロンプトは **financial_optimized_prompt.txt** に、Iterationごとの
プロンプト・評価結果・スコア履歴・Arizeメタプロンプトは
**financial_results/** に保存されます。

## 実験2：金融ニュース要約の結果

金融記事20件を対象に3回評価しました。GPT-4o Evaluatorは各要約を
次の3基準で1〜5点に評価し、具体的な自然言語フィードバックを生成しました。

- 主要な財務数値が含まれているか
- 不要な内容がなく簡潔か
- 原文にない内容を生成していないかという忠実性

### Iterationごとのスコア

| Iteration | 総合スコア | 財務数値 | 簡潔性 | 忠実性 |
|---|---:|---:|---:|---:|
| 1 | 83.67 | 3.85 | 3.95 | 4.75 |
| 2 | 83.67 | 4.00 | 3.95 | 4.60 |
| 3 | 83.33 | 3.95 | 3.90 | 4.65 |

Iteration 2では、主要な財務数値、経済的要因、影響と示唆を強調し、
根拠のない内容を除外する指示が追加されました。その結果、財務数値の
スコアは **3.85 → 4.00** に上昇しましたが、忠実性は
**4.75 → 4.60** に低下し、総合スコアは変わりませんでした。

Iteration 3では、金融市場と経済政策における重要な変化を強調する指示が
追加されましたが、総合スコアは **83.33** にわずかに低下しました。

### 改善しなかった原因

今回の実験では、財務数値を強調する指示をすべての記事に同じように適用した
ことで、副作用が発生しました。

- 数値の多い記事では、周辺の数値や企業ニュースまで過剰に列挙した
- 数値のない記事でも、財務指標が存在するかのように構造化した
- 情報量の増加により、簡潔性と忠実性が低下した
- Evaluatorは原文に数値がない場合でも財務数値のスコアを低く付けた

たとえば、あるサンプルは **66.67 → 46.67 → 73.33**、別のサンプルは
**66.67 → 66.67 → 40.00** と変化しました。全体平均では小さな変化でも、
個別の記事ではプロンプト変更によって大幅な性能低下が発生しました。

Iteration 1と2が同じ最高スコアだったため、同点の場合に先に評価された
プロンプトを維持する現在のロジックでは、初期プロンプトが最終結果として
選択されました。

~~~text
Summarize the following financial news article:

Article: {article}

Summary:
~~~

したがって、今回の金融要約実験の結論は「プロンプトが改善した」ではなく、
財務数値をより多く含めるよう最適化された一方、他の評価基準との
トレードオフにより、総合性能の向上には至らなかったということです。

Iterationごとのプロンプトと集計スコアは
[artifacts/financial/](artifacts/financial/)で確認できます。

## 2つの実験から得られた結論

| 区分 | 感情分類 | 金融ニュース要約 |
|---|---|---|
| 評価方法 | 正解ラベルとの直接比較 | GPT-4o Evaluator |
| フィードバック生成 | Pythonの比較ロジック | Evaluator LLM |
| 主な改善 | 文全体の感情を判断するルールとFew-shot例 | 財務数値と忠実性に関するルール |
| 主な限界 | 同じデータの再利用と過学習 | Evaluatorのバイアスと評価基準間のトレードオフ |
| 確認できた結果 | 具体的な改善プロンプトを生成 | 総合スコアは改善しなかった |

Prompt Learningは性能向上を保証するツールではなく、失敗原因を自然言語の
ルールへ変換し、プロンプトの修正プロセスを自動化するツールに近いものです。
特に自由形式の生成タスクでは、Optimizerより先に、Evaluatorの判断が
人間の判断と一致するかを検証する必要があります。

信頼できる実験にするため、今後は次の改善が必要です。

- Train、Validation、Testデータの分離
- 原文に財務数値がない場合の **not_applicable** 処理
- 評価基準ごとの独立したEvaluator
- 一部サンプルにおける人手評価とLLM評価の比較
- スコアが低下したプロンプトを適用しないロールバック
- APIコスト、トークン数、処理時間の記録

## 公開している実験成果物

再現性と結果比較のため、公開可能なプロンプトと集計スコアは
**artifacts/** に保存しています。

~~~text
artifacts/
├── emotion/
│   └── optimized_prompt.txt
└── financial/
    ├── optimized_prompt.txt
    ├── iteration_1_prompt.txt
    ├── iteration_2_prompt.txt
    ├── iteration_3_prompt.txt
    └── iteration_history.csv
~~~

記事原文と参照要約を含む **iteration_N_results.csv**、およびArizeが生成した
メタプロンプトは、データの再配布を避けるためリポジトリから除外しています。

これは結果を隠すためではありません。リポジトリではIterationごとの
プロンプトと集計スコアを公開し、実験プロセスと結論を確認できるように
しています。除外しているのは、第三者の記事全文を重複保存した生の実行
ファイルのみです。
