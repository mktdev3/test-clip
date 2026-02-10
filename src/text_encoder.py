"""
Text Encoder for CLIP Model

このモジュールは、CLIPモデルを使用してテキストをベクトル化するTextEncoderクラスを提供します。
"""

import logging
import unicodedata
from typing import List, Union
import numpy as np
import torch

from .clip_model import CLIPModel


# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextEncoder:
    """
    テキストエンコーディングクラス
    
    CLIPモデルを使用してテキストをベクトル化します。
    単一テキストとバッチ処理の両方に対応しています。
    """
    
    def __init__(self, clip_model: CLIPModel):
        """
        TextEncoderを初期化します。
        
        Args:
            clip_model: CLIPModelインスタンス
        
        Raises:
            ValueError: clip_modelが無効な場合
        """
        if not isinstance(clip_model, CLIPModel):
            raise ValueError("clip_modelはCLIPModelのインスタンスである必要があります")
        
        self.clip_model = clip_model
        logger.info("TextEncoderを初期化しました")

    def _normalize_text(self, text: str) -> str:
        """
        テキストを正規化します（NFKC正規化）。
        """
        if not isinstance(text, str):
             raise ValueError(f"テキストは文字列である必要があります: {type(text)}")
        
        text = text.strip()
        if not text:
            raise ValueError("テキストが空です")
            
        return unicodedata.normalize('NFKC', text)

    def encode_text(self, text: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        テキスト（またはテキストのリスト）をエンコードし、ベクトルを返します。
        リストの場合はバッチ処理を行います。
        
        Args:
            text: テキストまたはテキストのリスト
            batch_size: バッチ処理時のバッチサイズ
            
        Returns:
            テキストの特徴ベクトル
            - 単一テキストの場合: shape (512,)
            - リストの場合: shape (N, 512)
            
        Raises:
            ValueError: 入力が不正な場合
        """
        is_single = isinstance(text, str)
        if is_single:
            texts = [text]
        elif isinstance(text, list):
            texts = text
        else:
            raise ValueError(f"入力は文字列または文字列のリストである必要があります: {type(text)}")
            
        # Normalize
        try:
            normalized_texts = [self._normalize_text(t) for t in texts]
        except ValueError as e:
            logger.error(f"テキストの正規化に失敗しました: {e}")
            raise
        
        vectors = self._batch_encode(normalized_texts, batch_size)
        
        if is_single:
            return vectors.squeeze()
        return vectors

    def _batch_encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        内部メソッド：バッチ処理でエンコード
        """
        if not texts:
            raise ValueError("テキストリストが空です")
            
        all_vectors = []
        
        logger.info(f"{len(texts)}件のテキストをエンコードしています（バッチサイズ: {batch_size}）")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            try:
                inputs = self.clip_model.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
            except Exception as e:
                logger.error(f"トークナイズに失敗しました: {e}")
                raise RuntimeError(f"トークナイズエラー: {e}") from e
            
            # Move to device
            inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
            
            # Encode
            try:
                with torch.no_grad():
                    if hasattr(self.clip_model.model, 'text_model') and hasattr(self.clip_model.model, 'text_projection'):
                        outputs = self.clip_model.model.text_model(**inputs)
                        if hasattr(outputs, 'last_hidden_state'):
                            pooled = outputs.last_hidden_state[:, 0]
                        elif hasattr(outputs, 'pooler_output'):
                            pooled = outputs.pooler_output
                        else:
                            raise RuntimeError("text_modelの出力から特徴を取得できません")

                        features = self.clip_model.model.text_projection(pooled)
                    else:
                        output = self.clip_model.model.get_text_features(**inputs)

                        # Handle case where output is BaseModelOutputWithPooling (e.g. Rinna model)
                        if hasattr(output, 'pooler_output'):
                            pooled = output.pooler_output
                            if hasattr(self.clip_model.model, 'text_projection'):
                                projection = self.clip_model.model.text_projection
                                in_features = getattr(projection, 'in_features', None)
                                out_features = getattr(projection, 'out_features', None)
                                if in_features is None and hasattr(projection, 'weight'):
                                    in_features = projection.weight.shape[1]
                                    out_features = projection.weight.shape[0]
                                if in_features is not None and pooled.shape[-1] == in_features:
                                    features = projection(pooled)
                                elif out_features is not None and pooled.shape[-1] == out_features:
                                    # すでに射影済みの可能性があるため、そのまま使用
                                    features = pooled
                                else:
                                    logger.warning(
                                        "text_projectionの次元が不一致のため、pooler_outputをそのまま使用します: "
                                        f"pooler_dim={pooled.shape[-1]}, projection_in={in_features}, projection_out={out_features}"
                                    )
                                    features = pooled
                            else:
                                features = pooled
                        else:
                            features = output

                # Normalize
                features = features / features.norm(dim=-1, keepdim=True)

            except Exception as e:
                logger.error(f"エンコードに失敗しました: {e}")
                raise RuntimeError(f"エンコードエラー: {e}") from e
                
            batch_vectors = features.cpu().numpy()
            all_vectors.append(batch_vectors)
            
        result = np.vstack(all_vectors)
        logger.info(f"エンコード完了: {result.shape}")
        
        return result
