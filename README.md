# CLIP画像検索システム

日本語対応のCLIPモデルを使用した画像検索システム

## 概要

このプロジェクトは、日本語のテキストクエリで画像を検索できるシステムです。rinnaの日本語CLIPモデルを使用し、将来的にはベクトルデータベースによる大規模検索に対応します。

## 特徴

- 🇯🇵 **日本語対応**: rinna/japanese-clip-vit-b-16モデルを使用
- 🔍 **マルチモーダル検索**: テキストから画像を検索
- 🚀 **段階的スケーリング**: NumPy → Faiss → ベクトルDB
- 🐳 **Docker対応**: 環境構築を簡素化

## クイックスタート

### 必要要件

- Python 3.10以上
- CUDA対応GPU（推奨）

### インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd test-clip

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 使用方法

```bash
# インデックスの作成
python scripts/build_index.py

# CLI検索
python scripts/search_cli.py "猫の写真"

# Web UI（Phase 2以降）
streamlit run ui/app.py
```

## プロジェクト構成

```
test-clip/
├── docs/              # ドキュメント
│   └── specification.md
├── images/            # 画像データ（7枚のサンプル画像）
├── src/               # ソースコード
├── scripts/           # スクリプト
├── tests/             # テスト
└── requirements.txt   # 依存パッケージ
```

## 技術スタック

- **CLIPモデル**: rinna/japanese-clip-vit-b-16
- **深層学習**: PyTorch, Transformers
- **ベクトル検索**: NumPy → Faiss → Qdrant（段階的）
- **API**: FastAPI（Phase 2以降）
- **UI**: Streamlit（Phase 2以降）
- **コンテナ**: Docker（Phase 2以降）

## 開発ロードマップ

### Phase 1: プロトタイプ（現在）
- [x] プロジェクト構成
- [ ] 基本的な画像エンコーディング
- [ ] テキスト検索機能
- [ ] CLIインターフェース

### Phase 2: 最適化とUI
- [ ] Faiss導入
- [ ] Streamlit UI
- [ ] Docker化

### Phase 3: スケーリング
- [ ] ベクトルDB（Qdrant）導入
- [ ] 本番環境対応

## ドキュメント

詳細な仕様は [docs/specification.md](docs/specification.md) を参照してください。

## サンプル画像

`images/` フォルダには以下のサンプル画像が含まれています:
- 猫の写真
- りんごの写真
- 鳩の写真
- その他の画像

## ベクトルデータベースについて

### 現状の判断
現在の画像数（7枚）では、ベクトルデータベースは**不要**です。NumPyベースの実装で十分な性能が得られます。

### 導入タイミング
- **1,000枚以上**: Faissの導入を検討
- **10,000枚以上**: ベクトルDB（Qdrant, Milvus）の導入を検討

## ライセンス

このプロジェクトは [LICENSE](LICENSE) に基づいています。

## 参考資料

- [rinna/japanese-clip-vit-b-16](https://huggingface.co/rinna/japanese-clip-vit-b-16)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Qdrant](https://qdrant.tech/)
- [Faiss](https://github.com/facebookresearch/faiss)

---

**作成日**: 2026-02-10
