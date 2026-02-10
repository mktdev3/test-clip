"""
Image Encoder for CLIP Model

このモジュールは、CLIPモデルを使用して画像をベクトル化するImageEncoderクラスを提供します。
"""

import os
import logging
from typing import List, Union
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from .clip_model import CLIPModel


# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageEncoder:
    """
    画像エンコーディングクラス
    
    CLIPモデルを使用して画像をベクトル化します。
    単一画像とバッチ処理の両方に対応しています。
    """
    
    # サポートされている画像形式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}
    
    def __init__(self, clip_model: CLIPModel):
        """
        ImageEncoderを初期化します。
        
        Args:
            clip_model: CLIPModelインスタンス
        
        Raises:
            ValueError: clip_modelが無効な場合
        """
        if not isinstance(clip_model, CLIPModel):
            raise ValueError("clip_modelはCLIPModelのインスタンスである必要があります")
        
        self.clip_model = clip_model
        logger.info("ImageEncoderを初期化しました")
    
    def _load_and_preprocess(self, image_path: Union[str, Path]) -> Image.Image:
        """
        画像を読み込み、検証します。
        
        Args:
            image_path: 画像ファイルのパス
        
        Returns:
            PIL Image オブジェクト
        
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: サポートされていない形式の場合
            IOError: 画像の読み込みに失敗した場合
        """
        image_path = Path(image_path)
        
        # ファイルの存在確認
        if not image_path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        
        # 形式の確認
        if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"サポートされていない画像形式です: {image_path.suffix}\n"
                f"サポート形式: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # 画像の読み込み
        try:
            image = Image.open(image_path)
            # RGBモードに変換（RGBA等の場合に対応）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise IOError(f"画像の読み込みに失敗しました: {image_path}\n{str(e)}") from e
    
    def encode_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        単一画像をエンコードし、ベクトルを返します。
        
        Args:
            image_path: 画像ファイルのパス
        
        Returns:
            画像の特徴ベクトル（shape: (512,)）
        
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: サポートされていない形式の場合
            IOError: 画像の読み込みに失敗した場合
        """
        logger.info(f"画像をエンコードしています: {image_path}")
        
        # 画像の読み込みと検証
        image = self._load_and_preprocess(image_path)
        
        # 前処理
        inputs = self.clip_model.processor(images=image, return_tensors="pt")
        
        # デバイスへの転送
        inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
        
        # 推論実行
        with torch.no_grad():
            if hasattr(self.clip_model.model, 'vision_model') and hasattr(self.clip_model.model, 'visual_projection'):
                outputs = self.clip_model.model.vision_model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    pooled = outputs.last_hidden_state[:, 0]
                elif hasattr(outputs, 'pooler_output'):
                    pooled = outputs.pooler_output
                else:
                    raise RuntimeError("vision_modelの出力から特徴を取得できません")

                features = self.clip_model.model.visual_projection(pooled)
            else:
                features = self.clip_model.model.get_image_features(**inputs)

            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
        
        # NumPy配列に変換して返す
        vector = features.cpu().numpy().squeeze()
        logger.info(f"エンコード完了: ベクトル次元 {vector.shape}")
        
        return vector
    
    def encode_images_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        複数の画像をバッチ処理でエンコードします。
        
        Args:
            image_paths: 画像ファイルパスのリスト
            batch_size: バッチサイズ（デフォルト: 8）
        
        Returns:
            画像の特徴ベクトル配列（shape: (N, 512)）
        
        Raises:
            ValueError: image_pathsが空の場合、またはbatch_sizeが無効な場合
            FileNotFoundError: ファイルが存在しない場合
            IOError: 画像の読み込みに失敗した場合
        """
        if not image_paths:
            raise ValueError("image_pathsは空ではない必要があります")
        
        if batch_size <= 0:
            raise ValueError(f"batch_sizeは正の整数である必要があります: {batch_size}")
        
        logger.info(f"{len(image_paths)}枚の画像をバッチ処理でエンコードしています（バッチサイズ: {batch_size}）")
        
        all_vectors = []
        
        # バッチ処理
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            logger.info(f"バッチ {i // batch_size + 1}/{(len(image_paths) - 1) // batch_size + 1} を処理中")
            
            # 画像の読み込み
            images = [self._load_and_preprocess(path) for path in batch_paths]
            
            # 前処理
            inputs = self.clip_model.processor(images=images, return_tensors="pt")
            
            # デバイスへの転送
            inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
            
            # 推論実行
            with torch.no_grad():
                if hasattr(self.clip_model.model, 'vision_model') and hasattr(self.clip_model.model, 'visual_projection'):
                    outputs = self.clip_model.model.vision_model(**inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        pooled = outputs.last_hidden_state[:, 0]
                    elif hasattr(outputs, 'pooler_output'):
                        pooled = outputs.pooler_output
                    else:
                        raise RuntimeError("vision_modelの出力から特徴を取得できません")

                    features = self.clip_model.model.visual_projection(pooled)
                else:
                    features = self.clip_model.model.get_image_features(**inputs)

                # Normalize
                features = features / features.norm(dim=-1, keepdim=True)
            
            # デバッグログと互換性対応
            if not isinstance(features, torch.Tensor):
                if hasattr(features, 'image_embeds'):
                    features = features.image_embeds
                elif hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                else:
                    logger.warning("Could not identify tensor attribute, using features as is.")

            # NumPy配列に変換
            batch_vectors = features.cpu().numpy()
            all_vectors.append(batch_vectors)
        
        # すべてのバッチを結合
        result = np.vstack(all_vectors)
        logger.info(f"バッチ処理完了: {result.shape}")
        
        return result
    
    def __repr__(self) -> str:
        """
        オブジェクトの文字列表現を返します。
        """
        return f"ImageEncoder(clip_model={self.clip_model})"
