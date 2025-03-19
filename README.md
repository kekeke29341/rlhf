# LLMアライメント手法比較実験

## 概要

このプロジェクトは、大規模言語モデル（LLM）のアライメント手法における「生成・検証ギャップ」の影響を検証するシミュレーション実験を実装したものです。特に、「報酬モデル + 強化学習」の2段階アプローチ（RLHF）と直接最適化手法（DPO）の性能差を、生成・検証ギャップの有無によって比較します。

## 背景理論

### LLMのアライメント手法

現在のLLMアライメントには主に2つのアプローチがあります：

1. **2段階アプローチ（RLHF: Reinforcement Learning from Human Feedback）**
   - まず人間の選好から報酬モデルを学習
   - その報酬モデルを使ってLLM（方策モデル）を強化学習で最適化

2. **直接最適化アプローチ（DPO: Direct Preference Optimization）**
   - 中間の報酬モデルを経由せず、人間の選好データから直接LLMを最適化

興味深いことに、同じデータを使っているにもかかわらず、2段階アプローチの方が性能が高いことが実証されています。

### 生成・検証ギャップ仮説

この現象を説明する鍵は「生成より検証の方が簡単」という原理にあるという仮説があります：

- **報酬モデル（検証）**: 「どの応答が良いか」を評価する比較的簡単なタスク
- **方策モデル（生成）**: 良い応答を生成するという複雑なタスク

この非対称性により、報酬モデルはより早く正確に「良い応答の特徴」を学習でき、方策モデルは報酬モデルが識別した「良い応答空間」内だけで探索を行うことで、効率的に最適化が可能になるという考え方です。

## 必要条件

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- tqdm

## インストール方法

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/llm-alignment-experiment.git
cd llm-alignment-experiment

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

```bash
# 実験を実行
python rlhf_dpo_comparison.py
```

## プロジェクト構成

主要なコンポーネントは以下のとおりです：

### 1. 環境シミュレーション

```python
class SimplifiedEnvironment:
    """簡易的なLLM環境をシミュレートするクラス"""
```

- 入力と出力の次元、生成・検証の難易度差を調整できる環境を提供
- `complexity_factor`パラメータで生成・検証の難易度差を制御（値が大きいほど生成が検証より難しい）

### 2. モデル実装

```python
class PolicyModel(nn.Module):
    """方策モデル (LLM本体を簡易的に表現)"""

class RewardModel(nn.Module):
    """報酬モデル (人間の選好を学習)"""
```

- `PolicyModel`: LLM本体に相当するモデル
- `RewardModel`: 人間の選好を学習する報酬モデル

### 3. 学習アルゴリズム

```python
def train_reward_model(...):
    """報酬モデルを訓練する関数"""

def train_policy_with_rlhf(...):
    """RLHFで方策モデルを訓練する関数"""

def train_policy_with_dpo(...):
    """Direct Preference Optimizationで方策モデルを訓練する関数"""
```

### 4. 実験設計

```python
def run_experiment(env_with_gap, env_without_gap, num_iterations=5):
    """生成・検証ギャップの有無による性能差を検証する実験"""
```

- 生成・検証ギャップあり（`complexity_factor > 1`）
- 生成・検証ギャップなし（`complexity_factor = 1`）
- 各条件下でRLHFとDPOの性能を比較

## 実験結果の解釈

実験では以下の結果が期待されます：

1. **生成・検証ギャップがある場合**
   - RLHF（2段階アプローチ）がDPO（直接最適化）よりも高い性能を示す

2. **生成・検証ギャップがない場合**
   - RLHFとDPOの性能差が小さくなる、または消失する

この結果は「生成より検証の方が簡単」という性質が、なぜ2段階アプローチが効果的なのかを説明する重要な要因であることを示唆します。

## 生成される出力例

実行時には以下の出力が生成されます：

1. 訓練過程のログ（報酬モデルとポリシーモデルの学習経過）
2. 各手法・条件における最終性能の比較
3. RLHFとDPOの性能差を示すグラフ（`rlhf_vs_dpo_comparison.png`）

## 実装上の注意点

- このコードは教育・研究目的のシミュレーションであり、実際のLLMよりもはるかに小規模なモデルを使用
- 現象の本質を理解するための簡略化されたモデルであり、実際のLLMアライメントの複雑さをすべて再現しているわけではない

## 関連研究

- **InstructGPT (OpenAI)**: RLHFを大規模に適用した先駆的研究
- **Constitutional AI (Anthropic)**: 人間の価値観に沿った応答を学習するフレームワーク
- **Direct Preference Optimization (Stanford)**: 報酬モデルを経由せず直接最適化する手法
- **KL-constrained Preference Optimization**: 事前学習モデルからの逸脱を制限しつつ選好を学習

## ライセンス

MIT

## 参考文献

1. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*.

2. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*.

3. Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems*.

4. Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., ... & Kaplan, J. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv preprint arXiv:2212.08073*.
