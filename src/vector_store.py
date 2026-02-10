"""
Vector Storage Module

ベクトルデータとメタデータの保存・読み込みを行うモジュールです。
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    ベクトルとメタデータを管理するクラス
    
    Attributes:
        vector_dir (Path): ベクトルデータの保存ディレクトリ
        metadata_dir (Path): メタデータの保存ディレクトリ
    """

    def __init__(self, vector_dir: Union[str, Path], metadata_dir: Union[str, Path]):
        """
        VectorStoreを初期化します。

        Args:
            vector_dir: ベクトル保存先ディレクトリパス
            metadata_dir: メタデータ保存先ディレクトリパス
        """
        self.vector_dir = Path(vector_dir)
        self.metadata_dir = Path(metadata_dir)

        # ディレクトリが存在しない場合は作成
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VectorStore initialized: vectors={self.vector_dir}, metadata={self.metadata_dir}")

    def save(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """
        ベクトルとメタデータを保存します。
        既存のファイルは上書きされます。

        Args:
            vectors: 保存するベクトルデータ (numpy array)
            metadata: 保存するメタデータ (list of dicts)

        Raises:
            ValueError: ベクトルとメタデータの数が一致しない場合
            IOError: 保存に失敗した場合
        """
        if len(vectors) != len(metadata):
            raise ValueError(f"Vector count ({len(vectors)}) does not match metadata count ({len(metadata)})")

        vector_path = self.vector_dir / "vectors.npy"
        metadata_path = self.metadata_dir / "metadata.json"

        try:
            # ベクトルの保存
            np.save(vector_path, vectors)
            logger.info(f"Saved vectors to {vector_path}")

            # メタデータの保存
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise IOError(f"Failed to save data: {e}") from e

    def load(self) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """
        ベクトルとメタデータを読み込みます。

        Returns:
            Tuple[np.ndarray, List[Dict]]: (vectors, metadata)
            ファイルが存在しない場合は (None, None) を返します。

        Raises:
            IOError: 読み込みに失敗した場合（ファイルが存在するのに読み込めない場合など）
        """
        vector_path = self.vector_dir / "vectors.npy"
        metadata_path = self.metadata_dir / "metadata.json"

        if not vector_path.exists() or not metadata_path.exists():
            logger.warning("Vectors or metadata file not found.")
            return None, None

        try:
            # ベクトルの読み込み
            vectors = np.load(vector_path)
            
            # メタデータの読み込み
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 整合性チェック
            if len(vectors) != len(metadata):
                logger.error(f"Data inconsistency: vectors={len(vectors)}, metadata={len(metadata)}")
                raise ValueError("Loaded vectors and metadata counts do not match")

            logger.info(f"Loaded {len(vectors)} vectors and metadata")
            return vectors, metadata

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise IOError(f"Failed to load data: {e}") from e
