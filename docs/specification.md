# CLIP画像検索システム 仕様書

## 1. プロジェクト概要

### 1.1 目的
画像とテキストのマルチモーダル検索を実現するシステムを構築する。日本語のクエリで画像を検索できるようにし、将来的にはベクトル検索による高速な類似画像検索を実装する。

### 1.2 背景
- OpenAIのCLIPモデルは日本語に対応していない
- 日本語での画像検索を実現するため、日本語対応のCLIPモデルが必要
- 大規模な画像データベースに対応するため、ベクトルデータベースの導入を検討

## 2. システム要件

### 2.1 機能要件

#### 2.1.1 画像エンコーディング
- `images/` フォルダ内の画像をCLIPモデルでベクトル化
- サポート画像形式: JPG, PNG
- 画像の前処理とリサイズ

#### 2.1.2 テキスト検索
- 日本語テキストクエリによる画像検索
- テキストのベクトル化
- コサイン類似度による画像ランキング

#### 2.1.3 ベクトル検索
- 大規模データセットに対応した高速検索
- 類似画像の効率的な取得
- ベクトルインデックスの永続化

### 2.2 非機能要件

#### 2.2.1 パフォーマンス
- 検索レスポンス時間: 1秒以内（1000画像まで）
- ベクトル化処理: バッチ処理対応

#### 2.2.2 スケーラビリティ
- 画像数の増加に対応可能な設計
- ベクトルDBによる水平スケーリング

#### 2.2.3 保守性
- Docker化による環境の統一
- 依存関係の明確化
- ログ出力とエラーハンドリング

## 3. 技術選定

### 3.1 CLIPモデル

#### 3.1.1 選定理由
| 項目 | OpenAI CLIP | rinna CLIP |
|------|-------------|------------|
| 日本語対応 | ❌ 非対応 | ✅ 対応 |
| モデルサイズ | 複数サイズ | ViT-B/16 |
| 学習データ | 英語中心 | 日本語画像-テキストペア |
| ライセンス | MIT | Apache 2.0 |

**選定**: **rinna/japanese-clip-vit-b-16**
- 日本語テキストに最適化
- 日本語の画像キャプションで学習済み
- Hugging Faceから簡単に利用可能

#### 3.1.2 モデル情報
```
モデル名: rinna/japanese-clip-vit-b-16
提供元: rinna株式会社
アーキテクチャ: Vision Transformer (ViT-B/16)
入力画像サイズ: 224x224
埋め込み次元: 512次元
```

### 3.2 ベクトルデータベース

#### 3.2.1 必要性の検討

**ベクトルDBが必要なケース:**
- ✅ 画像数が10,000枚以上
- ✅ リアルタイム検索が必要
- ✅ 複数ユーザーからの同時アクセス
- ✅ 定期的な画像追加・更新

**現状の評価:**
- 現在: `images/` フォルダに7枚のサンプル画像
- 初期段階: **ベクトルDBは不要**
- 推奨: NumPy/Faissでの実装から開始

#### 3.2.2 段階的な導入計画

**Phase 1: プロトタイプ（現在）**
```python
# NumPyベースの実装
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ベクトルをメモリに保持
image_vectors = np.array([...])
similarities = cosine_similarity(query_vector, image_vectors)
```

**Phase 2: 中規模（1,000-10,000枚）**
```python
# Faissによる高速化
import faiss

# インデックスの作成
index = faiss.IndexFlatIP(512)  # 内積検索
index.add(image_vectors)
distances, indices = index.search(query_vector, k=10)
```

**Phase 3: 大規模（10,000枚以上）**
- ベクトルDB導入を検討
- 候補: Qdrant, Milvus, Weaviate, Pinecone

#### 3.2.3 ベクトルDB比較表

| DB名 | Docker対応 | 日本語ドキュメント | スケーラビリティ | 学習コスト |
|------|-----------|------------------|----------------|----------|
| **Qdrant** | ✅ | △ | 高 | 低 |
| **Milvus** | ✅ | △ | 非常に高 | 中 |
| **Weaviate** | ✅ | ❌ | 高 | 中 |
| **Pinecone** | ☁️ クラウドのみ | △ | 非常に高 | 低 |
| **Faiss** | ✅ | ❌ | 中 | 低 |

**推奨**: 初期はFaiss、将来的にQdrantへ移行

### 3.3 Docker化

#### 3.3.1 Docker化の利点
- ✅ 環境の一貫性（開発・本番環境の統一）
- ✅ 依存関係の管理（PyTorch, transformersなど）
- ✅ デプロイの簡素化
- ✅ ベクトルDBとの統合が容易

#### 3.3.2 コンテナ構成案

```yaml
# docker-compose.yml の構成イメージ
services:
  clip-api:
    # CLIP推論サーバー
    - Python 3.10
    - PyTorch
    - transformers
    - FastAPI
  
  vector-db:  # Phase 3で追加
    # Qdrant または Milvus
    
  web-ui:  # オプション
    # 検索UIフロントエンド
```

## 4. システムアーキテクチャ

### 4.1 コンポーネント図

```
┌─────────────────────────────────────────────────┐
│                   ユーザー                        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              Web UI / API                        │
│           (FastAPI / Streamlit)                  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           CLIP Inference Engine                  │
│        (rinna/japanese-clip-vit-b-16)           │
│  ┌──────────────┐      ┌──────────────┐        │
│  │ Image Encoder│      │ Text Encoder │        │
│  └──────────────┘      └──────────────┘        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│          Vector Storage & Search                 │
│  Phase 1: NumPy                                  │
│  Phase 2: Faiss                                  │
│  Phase 3: Qdrant/Milvus                         │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│            Image Storage                         │
│          (images/ directory)                     │
└─────────────────────────────────────────────────┘
```

### 4.2 データフロー

#### 4.2.1 インデックス作成フロー
```
1. images/ フォルダから画像を読み込み
2. 画像を前処理（リサイズ、正規化）
3. CLIPの画像エンコーダーでベクトル化
4. ベクトルをストレージに保存
5. メタデータ（ファイル名、パス）を保存
```

#### 4.2.2 検索フロー
```
1. ユーザーが日本語テキストを入力
2. CLIPのテキストエンコーダーでベクトル化
3. ベクトルストレージで類似度検索
4. Top-K件の画像を取得
5. 結果を表示（画像 + スコア）
```

## 5. 実装計画

### 5.1 Phase 1: プロトタイプ開発（1-2週間）

#### 5.1.1 環境セットアップ
- [ ] Python仮想環境の作成
- [ ] 依存パッケージのインストール
  - `torch`
  - `transformers`
  - `Pillow`
  - `numpy`
  - `scikit-learn`

#### 5.1.2 基本機能実装
- [ ] 画像読み込みモジュール
- [ ] CLIP推論モジュール
- [ ] ベクトル検索モジュール（NumPyベース）
- [ ] CLIインターフェース

#### 5.1.3 テスト
- [ ] サンプル画像での動作確認
- [ ] 日本語クエリのテスト
  - 例: "猫", "りんご", "山", "鳩"



## 6. ディレクトリ構成

```
test-clip/
├── src/
│   ├── __init__.py
│   ├── clip_model.py      # CLIPモデルのラッパー
│   ├── image_encoder.py   # 画像エンコーディング
│   ├── text_encoder.py    # テキストエンコーディング
│   ├── vector_store.py    # ベクトルストレージ
│   └── search.py          # 検索ロジック
├── scripts/
│   ├── build_index.py     # インデックス作成
│   └── search_cli.py      # CLI検索ツール
├── images/                # 画像データ
│   ├── PB144461bakusan_TP_V.jpg
│   ├── PED_narandaapple2_TP_V.jpg
│   ├── TKL-sc-K1_S0064-bc_TP_V.jpg
│   ├── hatoPAUI2850-12793_TP_V.jpg
│   ├── kotesuPAR58672_TP_V.jpg
│   ├── nekocyanPAKE5286-484_TP_V.jpg
│   └── tsurutama_MKT30720059_TP_V.jpg
├── data/
│   ├── vectors/           # ベクトルデータ
│   └── metadata/          # メタデータ
├── docs/
│   ├── specification.md   # 本ドキュメント
│   └── issues/            # 実装タスク
├── requirements.txt
└── README.md
```

## 7. 技術仕様

### 7.1 依存パッケージ

```txt
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
Pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
```

### 7.2 環境変数

```bash
# .env
MODEL_NAME=rinna/japanese-clip-vit-b-16
IMAGE_DIR=./images
VECTOR_DIR=./data/vectors
DEVICE=cuda  # または cpu
BATCH_SIZE=32
TOP_K=10
```



## 8. テスト計画

### 8.1 ユニットテスト
- [ ] 画像エンコーディングの正確性
- [ ] テキストエンコーディングの正確性
- [ ] ベクトル類似度計算



### 8.2 テストケース例

| テストケース | 入力 | 期待結果 |
|------------|------|---------|
| 猫画像検索 | "猫" | nekocyanPAKE5286-484_TP_V.jpg が上位 |
| りんご検索 | "りんご" | PED_narandaapple2_TP_V.jpg が上位 |
| 鳩検索 | "鳥" または "鳩" | hatoPAUI2850-12793_TP_V.jpg が上位 |





## 9. 参考資料

### 9.1 モデル
- [rinna/japanese-clip-vit-b-16](https://huggingface.co/rinna/japanese-clip-vit-b-16)
- [OpenAI CLIP](https://github.com/openai/CLIP)

### 9.2 フレームワーク
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

**ドキュメントバージョン**: 1.0  
**作成日**: 2026-02-10  
**最終更新**: 2026-02-10
