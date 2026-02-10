"""
CLIP Model Wrapper for rinna/japanese-clip-vit-b-16

このモジュールは、rinna/japanese-clip-vit-b-16モデルを読み込むラッパークラスを提供します。
"""

import os
import logging
from typing import Optional
import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from dotenv import load_dotenv


# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLIPModel:
    """
    CLIPモデルのラッパークラス
    
    rinna/japanese-clip-vit-b-16モデルを読み込み、管理します。
    デバイス設定（CPU/GPU）、環境変数からの設定読み込み、
    エラーハンドリングを提供します。
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        CLIPモデルを初期化します。
        
        Args:
            model_name: モデル名（デフォルト: 環境変数MODEL_NAMEまたは'rinna/japanese-clip-vit-b-16'）
            device: デバイス（'cpu', 'cuda', または None で自動検出）
        
        Raises:
            ValueError: モデル名が無効な場合
            RuntimeError: モデルの読み込みに失敗した場合
        """
        # 環境変数の読み込み
        load_dotenv()
        
        # モデル名の設定
        self.model_name = model_name or os.getenv('MODEL_NAME', 'rinna/japanese-clip-vit-b-16')
        logger.info(f"モデル名: {self.model_name}")
        
        # デバイスの設定
        if device is None:
            device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        logger.info(f"デバイス: {self.device}")
        
        # モデルの読み込み
        try:
            logger.info("モデルを読み込んでいます...")
            self.model = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # デバイスへの転送
            self.model.to(self.device)
            logger.info("モデルの読み込みが完了しました")
            
        except Exception as e:
            error_msg = f"モデルの読み込みに失敗しました: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_model_info(self) -> dict:
        """
        モデル情報を取得します。
        
        Returns:
            モデル名、デバイス、パラメータ数を含む辞書
        """
        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            'model_name': self.model_name,
            'device': self.device,
            'parameters': param_count,
            'cuda_available': torch.cuda.is_available()
        }
    
    def __repr__(self) -> str:
        """
        オブジェクトの文字列表現を返します。
        """
        return f"CLIPModel(model_name='{self.model_name}', device='{self.device}')"
