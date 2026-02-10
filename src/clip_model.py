"""
CLIP Model Wrapper for rinna/japanese-clip-vit-b-16

このモジュールは、rinna/japanese-clip-vit-b-16モデルを読み込むラッパークラスを提供します。
"""

import os
import logging
from typing import Optional
import torch
from transformers import VisionTextDualEncoderModel, AutoTokenizer, AutoImageProcessor
from dotenv import load_dotenv


# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class _JapaneseClipTokenizer:
    def __init__(self, tokenizer, device: str, max_seq_len: int = 77):
        self._tokenizer = tokenizer
        self._device = device
        self._max_seq_len = max_seq_len

    def __call__(self, texts, **kwargs):
        import japanese_clip as ja_clip

        if isinstance(texts, str):
            texts = [texts]

        return ja_clip.tokenize(
            texts=texts,
            tokenizer=self._tokenizer,
            max_seq_len=self._max_seq_len,
            device=self._device
        )


class _JapaneseClipProcessor:
    def __init__(self, preprocess, device: str):
        self._preprocess = preprocess
        self._device = device

    def __call__(self, images=None, **kwargs):
        if images is None:
            raise ValueError("imagesが指定されていません")

        if not isinstance(images, list):
            images = [images]

        pixel_values = torch.stack([self._preprocess(img) for img in images]).to(self._device)
        return {"pixel_values": pixel_values}


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

            use_japanese_clip = os.getenv("USE_JAPANESE_CLIP", "1").lower() in {"1", "true", "yes"}
            if use_japanese_clip:
                try:
                    import japanese_clip as ja_clip

                    cache_dir = os.getenv("JAPANESE_CLIP_CACHE_DIR")
                    if cache_dir:
                        model, preprocess = ja_clip.load(self.model_name, cache_dir=cache_dir, device=self.device)
                    else:
                        model, preprocess = ja_clip.load(self.model_name, device=self.device)

                    tokenizer = ja_clip.load_tokenizer()

                    self.model = model
                    self.tokenizer = _JapaneseClipTokenizer(tokenizer, self.device)
                    self.processor = _JapaneseClipProcessor(preprocess, self.device)
                    self.is_japanese_clip = True

                    logger.info("japanese-clip 経由でモデルを読み込みました")
                    logger.info("モデルの読み込みが完了しました")
                    return
                except Exception as e:
                    logger.warning(f"japanese-clipの読み込みに失敗しました。transformersにフォールバックします: {e}")

            self.model = VisionTextDualEncoderModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            except OSError:
                logger.warning(f"Could not load image processor from {self.model_name}, falling back to 'openai/clip-vit-base-patch16'")
                self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

            self.is_japanese_clip = False

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
