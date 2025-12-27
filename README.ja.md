# 🛒 AI搭載ショッピングアシスタント

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Tests](https://img.shields.io/badge/tests-26%2F26%20passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**インテリジェントなeコマースカスタマーサービスのための本番環境対応の会話型AIシステム**

[機能](#-機能) • [デモ](#-デモ) • [インストール](#-インストール) • [使い方](#-使い方) • [アーキテクチャ](#-アーキテクチャ) • [コントリビューション](#-コントリビューション)

[English](README.md) | 日本語

</div>

---

## 📋 目次

- [概要](#-概要)
- [機能](#-機能)
- [デモ](#-デモ)
- [アーキテクチャ](#-アーキテクチャ)
- [技術スタック](#-技術スタック)
- [インストール](#-インストール)
- [使い方](#-使い方)
- [プロジェクト構造](#-プロジェクト構造)
- [テスト](#-テスト)
- [設定](#-設定)
- [APIドキュメント](#-apiドキュメント)
- [トラブルシューティング](#-トラブルシューティング)
- [コントリビューション](#-コントリビューション)
- [ライセンス](#-ライセンス)
- [謝辞](#-謝辞)

---

## 🎯 概要

このプロジェクトは、最先端のAI技術を使用してeコマースのカスタマーサービスを革新するインテリジェントなショッピングアシスタントを実装しています。**LangGraph**、**LangChain**、**Model Context Protocol (MCP)** で構築され、商品発見、カート管理、カスタマーサポートのためのシームレスな会話体験を提供します。

### 課題

従来のeコマースプラットフォームは以下の問題を抱えています：
- 自然言語を理解しない一般的な検索結果
- 商品発見と購入の間で分断されたショッピング体験
- 日常的な問い合わせで圧倒されるカスタマーサポートチーム
- 購入履歴に基づくパーソナライゼーションの欠如

### ソリューション

私たちのAIショッピングアシスタントは以下を提供します：
- 直感的な商品検索のための**自然言語理解**
- ショッピングセッションを記憶する**コンテキスト会話**
- 必要に応じた人間サポートへの**インテリジェントなエスカレーション**
- 商品情報とトレンドのための**リアルタイムWeb統合**
- パーソナライズされた推奨のための**購入履歴分析**

---

## ✨ 機能

### 🔍 インテリジェント商品検索

- **セマンティック検索**: 「健康的な朝食オプション」のような自然言語クエリを理解
- **ベクトル埋め込み**: 49,688商品にHuggingFace mxbai-embed-large-v1を使用
- **構造化フィルタリング**: 正確な結果のためのSQLライクなクエリ（部門、通路、価格）
- **購入履歴**: パーソナライズされた提案のために過去の注文を分析

### 🛒 スマートショッピングカート

- **数量管理**: 追加、削除、数量更新
- **セッション永続性**: 会話全体でカートを維持
- **リアルタイム更新**: カート操作の即座のフィードバック
- **チェックアウトフロー**: 完全な購入ワークフロー

### 🌐 Web検索統合 (MCP)

- **6つのBrave検索ツール**:
  - 一般的なWeb検索
  - ニュース記事
  - ビデオコンテンツ
  - 画像検索
  - ローカル結果
  - AI要約
- **ハイブリッドクエリ**: 商品カタログとリアルタイムWebデータを組み合わせ

### 🤝 カスタマーサポート

- **自動エスカレーション**: 返金リクエスト、苦情、アカウント問題を検出
- **Human-in-the-Loop**: 機密操作には監督者の承認が必要
- **コンテキスト保持**: サポートエージェントが完全な会話履歴にアクセス可能

### 📊 分析とインサイト

- **購入パターン**: 最も購入されたアイテム、再注文頻度
- **商品推奨**: 注文履歴に基づく
- **トレンド分析**: Web検索とカタログデータを組み合わせ

---

## 🎬 デモ

### 会話例

**商品発見**
```
👤 ユーザー: 「健康的な朝食オプションが欲しいです」
🤖 アシスタント: カタログから健康的な朝食アイテムをご紹介します：
   • オーガニックグラノーラ (ID: 18872) - シリアル通路
   • ギリシャヨーグルト (ID: 24531) - 乳製品通路
   • フレッシュブルーベリー (ID: 13176) - 新鮮な果物通路
   カートに追加しますか？
```

**カート管理**
```
👤 ユーザー: 「グラノーラを2箱カートに追加して」
🤖 アシスタント: オーガニックグラノーラ (ID: 18872) を2個カートに追加しました。

👤 ユーザー: 「カートを見せて」
🤖 アシスタント: カートの内容：
   • オーガニックグラノーラ (ID: 18872) × 2
   合計アイテム: 2
```

**Web検索 + 商品カタログ**
```
👤 ユーザー: 「キヌアの健康効果は？在庫ありますか？」
🤖 アシスタント: [Web検索] キヌアは9つの必須アミノ酸すべてを含む
   完全なタンパク質で、食物繊維が豊富でグルテンフリーです...
   
   [カタログ検索] はい！在庫があります：
   • オーガニックキヌア (ID: 39187) - 穀物通路
   • トリカラーキヌア (ID: 42156) - 穀物通路
```

**カスタマーサポート**
```
👤 ユーザー: 「前回の注文の返金をお願いしたい」
🤖 アシスタント: 返金に関してカスタマーサポートにおつなぎします。
   [サポートエージェントにエスカレート]
   
🎧 サポート: 返金のご要望ですね。監督者の承認を取得します...
   [人間の承認をリクエスト]
   
👨‍💼 監督者: [返金を承認]
🎧 サポート: 返金が承認されました。3〜5日以内に処理されます。
```

---

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                         ユーザーインターフェース                  │
│                    (Streamlit Webアプリ)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraphステートマシン                       │
│              (会話フローオーケストレーション)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AIエージェント (GPT-4o-mini)                   │
│              (自然言語理解)                                      │
└──────┬──────────────────┬──────────────────┬───────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌──────────────────┐
│   商品検索   │  │ ショッピング │  │   Web検索        │
│             │  │   カート     │  │   (MCP/Brave)    │
└─────────────┘  └─────────────┘  └──────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌──────────────────┐
│  ベクトルDB  │  │  インメモリ  │  │  Brave Search    │
│  (Chroma)   │  │   ストレージ │  │      API         │
│             │  │             │  │                  │
│ 49,688      │  │ セッション   │  │ リアルタイム     │
│ 商品        │  │ ベース      │  │ Webデータ        │
└─────────────┘  └─────────────┘  └──────────────────┘
```

### 主要コンポーネント

1. **LangGraphステートマシン**: 会話フローと状態遷移を管理
2. **AIエージェント**: 自然言語理解と生成のためのGPT-4o-mini
3. **ツールレイヤー**: 検索、カート、エスカレーションのためのモジュラーツール
4. **データレイヤー**: ベクトルDB (Chroma) + 構造化データ (Pandas)
5. **MCP統合**: Web検索のための外部ツールプロトコル

---

## 🛠️ 技術スタック

### コアフレームワーク
- **[LangChain](https://python.langchain.com/)** - LLM統合とツール管理
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - ステートフル会話オーケストレーション
- **[OpenAI GPT-4o-mini](https://openai.com/)** - 自然言語処理

### データと検索
- **[Chroma](https://www.trychroma.com/)** - セマンティック検索のためのベクトルデータベース
- **[HuggingFace Transformers](https://huggingface.co/)** - テキスト埋め込み (mxbai-embed-large-v1)
- **[Pandas](https://pandas.pydata.org/)** - データ操作とフィルタリング

### 統合
- **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/)** - 外部ツール統合
- **[Brave Search API](https://brave.com/search/api/)** - Web検索機能

### 開発
- **[Pydantic](https://pydantic.dev/)** - データ検証とスキーマ定義
- **[Pytest](https://pytest.org/)** - テストフレームワーク
- **[Streamlit](https://streamlit.io/)** - Webインターフェース
- **[Python-dotenv](https://pypi.org/project/python-dotenv/)** - 環境管理

---

## 📦 インストール

### 前提条件

- **Python 3.10+** ([ダウンロード](https://www.python.org/downloads/))
- **Node.js** ([ダウンロード](https://nodejs.org/)) - MCPサーバーに必要
- **OpenAI APIキー** ([取得](https://platform.openai.com/api-keys))
- **Brave Search APIキー** ([取得](https://brave.com/search/api/)) - オプション

### ステップ1: リポジトリのクローン

```bash
git clone https://github.com/yourusername/ai-shopping-assistant.git
cd ai-shopping-assistant
```

### ステップ2: 仮想環境の作成

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### ステップ3: 依存関係のインストール

```bash
pip install -r requirements.txt
```

### ステップ4: 環境変数の設定

プロジェクトルートに`.env`ファイルを作成：

```bash
# 必須
OPENAI_API_KEY=sk-your-openai-api-key-here

# オプション (Web検索用)
BRAVE_API_KEY=your-brave-api-key-here
```

### ステップ5: データセットのダウンロード

データセットはサイズ（約50MB）のためリポジトリに含まれていません。自動的にダウンロード：

```bash
python download_dataset.py
```

これにより：
- Google Driveから食料品店データセットをダウンロード
- `dataset/`フォルダに展開
- 必要なすべてのCSVファイルを検証

**データセットの内容:**
- `products.csv` - 49,688商品（名前、部門、通路）
- `orders.csv` - 340万件以上の注文記録
- `order_products__prior.csv` - 購入履歴
- `order_products__train.csv` - トレーニングデータ
- `departments.csv` - 21部門
- `aisles.csv` - 134通路

### ステップ6: ベクトルデータベースの構築

ベクトルデータベースもサイズのため含まれていません。ローカルで構築：

#### オプションA: Google Colab（推奨 - 10倍高速）

1. [Google Colab](https://colab.research.google.com)を開く
2. GPUを有効化: `ランタイム` → `ランタイムのタイプを変更` → `T4 GPU`
3. 実行:

```python
!git clone https://github.com/yourusername/ai-shopping-assistant.git
%cd ai-shopping-assistant
!pip install -r requirements.txt
!python download_dataset.py
!python src/build_vector_db.py
```

4. `vector_db/`フォルダをダウンロード
5. ローカルプロジェクトルートに配置

**所要時間: GPUで約5〜10分**

#### オプションB: ローカルマシン

```bash
python src/build_vector_db.py
```

**所要時間: CPUで約1〜2時間**

### ステップ7: インストールの確認

```bash
# テストを実行
pytest tests/ -v

# 表示されるはず: 26 passed
```

---

## 🚀 使い方

### Webインターフェース（推奨）

Streamlitアプリを起動：

```bash
streamlit run app.py
```

ブラウザで開く: **http://localhost:8501**

### コマンドラインインターフェース

```python
from src.conversation_runner import run_single_turn

# 単一ターン会話
result = run_single_turn(
    user_input="健康的なスナックが欲しい",
    thread_id="my-session-123"
)

print(result['response'])
```

### プログラマティック使用

```python
from src.graph import graph
from langchain_core.messages import HumanMessage

# マルチターン会話
config = {"configurable": {"thread_id": "session-456"}}

# 最初のメッセージ
state1 = graph.invoke(
    {"messages": [HumanMessage(content="オーガニック商品を見せて")]},
    config
)

# フォローアップ（コンテキストを維持）
state2 = graph.invoke(
    {"messages": [HumanMessage(content="最初のものをカートに追加して")]},
    config
)
```

---

## 📁 プロジェクト構造

```
ai-shopping-assistant/
│
├── src/                          # ソースコード
│   ├── assistants.py             # AIエージェント実装（販売とサポート）
│   ├── tools.py                  # ツール定義（検索、カート、エスカレーション）
│   ├── graph.py                  # LangGraph会話フロー
│   ├── state.py                  # 状態管理スキーマ
│   ├── prompts.py                # AIプロンプトと指示
│   ├── web_search_mcp.py         # Brave Search MCP統合
│   ├── build_vector_db.py        # ベクトルデータベースビルダー
│   ├── conversation_runner.py    # テストユーティリティ
│   └── __init__.py
│
├── tests/                        # テストスイート（100%カバレッジ）
│   ├── test_cart_and_schema.py   # カートとスキーマ検証テスト
│   ├── test_end_to_end.py        # 統合テスト
│   ├── test_graph.py             # グラフフローテスト
│   ├── test_sales_assistant.py   # 販売エージェントテスト
│   ├── test_structured_search.py # 構造化検索テスト
│   ├── test_tool_node.py         # ツール実行テスト
│   ├── test_vector_search.py     # ベクトル検索テスト
│   ├── test_web_search_mcp.py    # MCP統合テスト
│   └── __init__.py
│
├── dataset/                      # 商品カタログ（ダウンロード）
│   ├── products.csv              # 49,688商品
│   ├── orders.csv                # 注文履歴
│   ├── order_products__prior.csv # 購入記録
│   ├── order_products__train.csv # トレーニングデータ
│   ├── departments.csv           # 21部門
│   └── aisles.csv                # 134通路
│
├── vector_db/                    # Chromaベクトルデータベース（ローカル構築）
│   └── [生成されたファイル]
│
├── app.py                        # Streamlit Webインターフェース
├── requirements.txt              # Python依存関係
├── download_dataset.py           # データセットダウンローダースクリプト
├── .env                          # 環境変数（リポジトリに含まれない）
├── .gitignore                    # Git無視ルール
├── README.md                     # 英語版README
├── README.ja.md                  # このファイル
├── ASSIGNMENT.md                 # 元の課題指示
└── LICENSE                       # MITライセンス
```

---

## 🧪 テスト

### すべてのテストを実行

```bash
pytest tests/ -v
```

**期待される出力:**
```
26 passed in ~20s
```

### 特定のテストスイートを実行

```bash
# ベクトル検索テスト
pytest tests/test_vector_search.py -v

# 構造化検索テスト
pytest tests/test_structured_search.py -v

# カート機能テスト
pytest tests/test_cart_and_schema.py -v

# MCP統合テスト
pytest tests/test_web_search_mcp.py -v

# エンドツーエンドワークフローテスト
pytest tests/test_end_to_end.py -v
```

### テストカバレッジ

```bash
pytest tests/ --cov=src --cov-report=html
```

`htmlcov/index.html`を開いて詳細なカバレッジレポートを表示。

**現在のカバレッジ: 100%**

---

## ⚙️ 設定

### 環境変数

| 変数 | 必須 | 説明 |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ はい | GPT-4o-mini用のOpenAI APIキー |
| `BRAVE_API_KEY` | ❌ オプション | Web検索用のBrave Search APIキー |

### モデル設定

`src/assistants.py`を編集してLLMを変更：

```python
# 現在: GPT-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 代替: GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 代替: GPT-3.5-turbo
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
```

---

## 🐛 トラブルシューティング

### よくある問題

#### 1. "No module named 'langchain'"

**解決策:**
```bash
pip install -r requirements.txt
```

#### 2. "OPENAI_API_KEY not found"

**解決策:**
- プロジェクトルートに`.env`ファイルを作成
- 追加: `OPENAI_API_KEY=your-key-here`
- アプリケーションを再起動

#### 3. "Vector database not found"

**解決策:**
```bash
python src/build_vector_db.py
```

#### 4. "Dataset files missing"

**解決策:**
```bash
python download_dataset.py
```

#### 5. "Tests failing"

**解決策:**
```bash
# vector_dbが存在することを確認
python src/build_vector_db.py

# データセットが存在することを確認
python download_dataset.py

# テストを実行
pytest tests/ -v
```

#### 6. "MCP tools not loading"

**解決策:**
- Node.jsがインストールされているか確認: `node --version`
- `.env`でBRAVE_API_KEYを確認
- インターネット接続を確認

---

## 🤝 コントリビューション

コントリビューションを歓迎します！以下のガイドラインに従ってください：

### 開発セットアップ

1. リポジトリをフォーク
2. 機能ブランチを作成: `git checkout -b feature/amazing-feature`
3. 変更を加える
4. テストを実行: `pytest tests/ -v`
5. コミット: `git commit -m 'Add amazing feature'`
6. プッシュ: `git push origin feature/amazing-feature`
7. プルリクエストを開く

### コードスタイル

コードフォーマットには[Black](https://black.readthedocs.io/)を使用：

```bash
black --line-length=88 .
```

### テスト要件

- すべての新機能にはテストを含める必要があります
- 100%のテストカバレッジを維持
- PR承認前にすべてのテストが合格する必要があります

---

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

---

## 🙏 謝辞

### データセット
- **Instacart Market Basket Analysis** - 49,000以上の商品を含む食料品店データセット

### 技術
- **LangChain & LangGraph** - 会話オーケストレーションフレームワーク
- **OpenAI** - 自然言語理解のためのGPTモデル
- **Brave Search** - Web検索APIとMCP統合
- **HuggingFace** - テキスト埋め込みモデル
- **Chroma** - ベクトルデータベース

### インスピレーション
- このプロジェクトはLLMエージェントコースの一環として開発されました
- コースインストラクターとコミュニティに感謝します

---

## 📞 連絡先とサポート

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-shopping-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-shopping-assistant/discussions)
- **Email**: joel_solaeche@hotmail.com

---

## 🗺️ ロードマップ

### 現在のバージョン (v1.0)
- ✅ セマンティック商品検索
- ✅ ショッピングカート管理
- ✅ カスタマーサポートエスカレーション
- ✅ Web検索統合 (MCP)
- ✅ 100%テストカバレッジ

### 将来の機能強化 (v2.0)
- [ ] 多言語サポート
- [ ] 音声インターフェース
- [ ] 商品推奨MLモデル
- [ ] 注文追跡統合
- [ ] 決済処理
- [ ] モバイルアプリ
- [ ] 分析ダッシュボード

---

<div align="center">

**LangGraph、LangChain、MCPで❤️を込めて構築**

⭐ 役に立ったらこのリポジトリにスターをつけてください！

[バグ報告](https://github.com/yourusername/ai-shopping-assistant/issues) • [機能リクエスト](https://github.com/yourusername/ai-shopping-assistant/issues)

</div>
