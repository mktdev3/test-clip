"""
Unit tests for ImageEncoder

ImageEncoderクラスのユニットテストを提供します。
"""

import os
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

os.environ.setdefault('USE_JAPANESE_CLIP', '0')

from src.clip_model import CLIPModel
from src.image_encoder import ImageEncoder


class TestImageEncoder:
    """ImageEncoderクラスのテストスイート"""
    
    @pytest.fixture
    def mock_clip_model(self):
        """モックCLIPModelを作成するフィクスチャ"""
        with patch('src.clip_model.VisionTextDualEncoderModel.from_pretrained') as mock_model, \
             patch('src.clip_model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.clip_model.AutoImageProcessor.from_pretrained') as mock_processor:
            
            # モックモデルの設定
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_tokenizer.return_value = MagicMock()
            
            # プロセッサーのモック設定（動的に戻り値を生成）
            mock_processor_instance = MagicMock()
            
            def processor_side_effect(images=None, return_tensors=None, **kwargs):
                if isinstance(images, list):
                    batch_size = len(images)
                else:
                    batch_size = 1
                return {
                    'pixel_values': torch.randn(batch_size, 3, 224, 224)
                }
            
            mock_processor_instance.side_effect = processor_side_effect
            mock_processor.return_value = mock_processor_instance
            
            # get_image_featuresのモック設定（入力バッチサイズに応じたベクトルを返す）
            def get_features_side_effect(**kwargs):
                if 'pixel_values' in kwargs:
                    batch_size = kwargs['pixel_values'].shape[0]
                else:
                    batch_size = 1
                return torch.randn(batch_size, 512)
                
            mock_model_instance.get_image_features.side_effect = get_features_side_effect
            
            # CLIPModelインスタンスを作成
            clip_model = CLIPModel(device='cpu')
            
            yield clip_model
    
    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """テスト用のサンプル画像を作成するフィクスチャ"""
        # 簡単なテスト画像を作成
        image = Image.new('RGB', (100, 100), color='red')
        image_path = tmp_path / "test_image.jpg"
        image.save(image_path)
        return image_path
    
    @pytest.fixture
    def real_image_path(self):
        """実際のサンプル画像パスを返すフィクスチャ"""
        # images/フォルダ内の実際の画像を使用
        image_dir = Path(__file__).parent.parent / "images"
        image_files = list(image_dir.glob("*.jpg"))
        if image_files:
            return image_files[0]
        return None
    
    def test_initialization(self, mock_clip_model):
        """ImageEncoderの初期化テスト"""
        encoder = ImageEncoder(mock_clip_model)
        assert encoder.clip_model == mock_clip_model
        assert encoder.SUPPORTED_FORMATS == {'.jpg', '.jpeg', '.png'}
    
    def test_initialization_invalid_model(self):
        """無効なモデルでの初期化テスト"""
        with pytest.raises(ValueError) as exc_info:
            ImageEncoder("invalid_model")
        assert "CLIPModelのインスタンスである必要があります" in str(exc_info.value)
    
    def test_encode_single_image(self, mock_clip_model, sample_image_path):
        """単一画像のエンコードテスト"""
        encoder = ImageEncoder(mock_clip_model)
        vector = encoder.encode_image(sample_image_path)
        
        # ベクトルの検証
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (512,)
        assert vector.dtype == np.float32 or vector.dtype == np.float64
    
    def test_encode_image_file_not_found(self, mock_clip_model):
        """存在しないファイルのエラーハンドリングテスト"""
        encoder = ImageEncoder(mock_clip_model)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            encoder.encode_image("nonexistent_file.jpg")
        assert "画像ファイルが見つかりません" in str(exc_info.value)
    
    def test_encode_image_invalid_format(self, mock_clip_model, tmp_path):
        """非対応形式のエラーハンドリングテスト"""
        encoder = ImageEncoder(mock_clip_model)
        
        # .txt ファイルを作成
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not an image")
        
        with pytest.raises(ValueError) as exc_info:
            encoder.encode_image(invalid_file)
        assert "サポートされていない画像形式です" in str(exc_info.value)
    
    def test_encode_batch_images(self, mock_clip_model, tmp_path):
        """バッチ処理のテスト"""
        encoder = ImageEncoder(mock_clip_model)
        
        # 複数のテスト画像を作成
        image_paths = []
        for i in range(5):
            image = Image.new('RGB', (100, 100), color='blue')
            image_path = tmp_path / f"test_image_{i}.jpg"
            image.save(image_path)
            image_paths.append(image_path)
        
        # バッチ処理実行
        vectors = encoder.encode_images_batch(image_paths, batch_size=2)
        
        # 結果の検証
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (5, 512)
    
    def test_encode_batch_empty_list(self, mock_clip_model):
        """空のリストでのバッチ処理エラーテスト"""
        encoder = ImageEncoder(mock_clip_model)
        
        with pytest.raises(ValueError) as exc_info:
            encoder.encode_images_batch([])
        assert "空ではない必要があります" in str(exc_info.value)
    
    def test_encode_batch_invalid_batch_size(self, mock_clip_model, sample_image_path):
        """無効なバッチサイズのエラーテスト"""
        encoder = ImageEncoder(mock_clip_model)
        
        with pytest.raises(ValueError) as exc_info:
            encoder.encode_images_batch([sample_image_path], batch_size=0)
        assert "正の整数である必要があります" in str(exc_info.value)
    
    def test_vector_dimension(self, mock_clip_model, sample_image_path):
        """出力ベクトルが512次元であることを確認"""
        encoder = ImageEncoder(mock_clip_model)
        vector = encoder.encode_image(sample_image_path)
        
        assert vector.shape == (512,), f"Expected shape (512,), got {vector.shape}"
    
    def test_supported_formats_jpg(self, mock_clip_model, tmp_path):
        """JPG形式のサポート確認"""
        encoder = ImageEncoder(mock_clip_model)
        
        # JPG画像を作成
        image = Image.new('RGB', (100, 100), color='green')
        image_path = tmp_path / "test.jpg"
        image.save(image_path)
        
        # エンコードが成功することを確認
        vector = encoder.encode_image(image_path)
        assert vector.shape == (512,)
    
    def test_supported_formats_png(self, mock_clip_model, tmp_path):
        """PNG形式のサポート確認"""
        encoder = ImageEncoder(mock_clip_model)
        
        # PNG画像を作成
        image = Image.new('RGB', (100, 100), color='yellow')
        image_path = tmp_path / "test.png"
        image.save(image_path)
        
        # エンコードが成功することを確認
        vector = encoder.encode_image(image_path)
        assert vector.shape == (512,)
    
    def test_corrupted_image(self, mock_clip_model, tmp_path):
        """破損画像のエラーハンドリングテスト"""
        encoder = ImageEncoder(mock_clip_model)
        
        # 破損した画像ファイルを作成（実際には画像ではないデータ）
        corrupted_file = tmp_path / "corrupted.jpg"
        corrupted_file.write_bytes(b"This is not a valid image file")
        
        with pytest.raises(IOError) as exc_info:
            encoder.encode_image(corrupted_file)
        assert "画像の読み込みに失敗しました" in str(exc_info.value)
    
    def test_rgba_to_rgb_conversion(self, mock_clip_model, tmp_path):
        """RGBA画像のRGB変換テスト"""
        encoder = ImageEncoder(mock_clip_model)
        
        # RGBA画像を作成
        image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        image_path = tmp_path / "test_rgba.png"
        image.save(image_path)
        
        # エンコードが成功することを確認
        vector = encoder.encode_image(image_path)
        assert vector.shape == (512,)
    
    def test_repr(self, mock_clip_model):
        """__repr__メソッドのテスト"""
        encoder = ImageEncoder(mock_clip_model)
        repr_str = repr(encoder)
        
        assert 'ImageEncoder' in repr_str
        assert 'CLIPModel' in repr_str
