"""
Search Engine Module

テキストクエリ（ベクトル）を用いて画像を検索する機能を提供します。
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .vector_store import VectorStore

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchEngine:
    """
    画像検索を行うクラス
    
    Attributes:
        vector_store (VectorStore): ベクトルストアのインスタンス
        vectors (np.ndarray): ロードされたベクトルデータ
        metadata (List[Dict]): ロードされたメタデータ
    """

    def __init__(self, vector_store: VectorStore):
        """
        SearchEngineを初期化します。

        Args:
            vector_store: 初期化済みのVectorStoreインスタンス
        """
        self.vector_store = vector_store
        self.vectors: Optional[np.ndarray] = None
        self.metadata: Optional[List[Dict]] = None
        
        self._load_data()

    def _load_data(self):
        """ベクトルストアからデータをロードします。"""
        self.vectors, self.metadata = self.vector_store.load()
        if self.vectors is None or self.metadata is None:
            logger.warning("No data found in vector store. Search will not function properly.")

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Union[str, float]]]:
        """
        クエリベクトルに類似する画像を検索します。

        Args:
            query_vector: 検索クエリのベクトル (1次元または2次元numpy array)
            top_k: 取得する上位件数

        Returns:
            List[Dict]: 検索結果のリスト。各要素は {'image_path': str, 'score': float}
            
        Raises:
            ValueError: ベクトルデータがロードされていない、またはクエリの次元が不正な場合
        """
        if self.vectors is None or self.metadata is None:
            # 再度ロードを試みる（データが後から追加された場合など）
            self._load_data()
            if self.vectors is None or self.metadata is None:
                raise ValueError("Vector data is not loaded.")

        # クエリベクトルの形状確認
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        elif query_vector.ndim != 2:
             raise ValueError(f"Invalid query vector shape: {query_vector.shape}")

        if query_vector.shape[1] != self.vectors.shape[1]:
             raise ValueError(f"Dimension mismatch: query={query_vector.shape[1]}, index={self.vectors.shape[1]}")

        try:
            # コサイン類似度の計算
            # query_vector: (1, dim), self.vectors: (n, dim) -> similarities: (1, n)
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            
            # スコア順にソート（降順）
            # argsortは昇順なので、[::-1]で降順にし、[:top_k]で上位を取得
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                # スコアが負になることは理論上ないが（cosine similarityは-1~1）、念のため
                score = float(similarities[idx])
                
                result = {
                    'image_path': self.metadata[idx].get('path', 'unknown'),
                    'score': score
                }
                # 必要に応じて他のメタデータも含めることが可能
                results.append(result)
                
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed: {e}") from e
