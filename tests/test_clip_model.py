"""
Unit tests for CLIPModel wrapper

CLIPModelクラスのユニットテストを提供します。
"""

import os
import pytest
import torch
from unittest.mock import patch, MagicMock
os.environ.setdefault('USE_JAPANESE_CLIP', '0')

from src.clip_model import CLIPModel


class TestCLIPModel:
    """CLIPModelクラスのテストスイート"""
    
    def test_model_initialization_default(self):
        """デフォルト設定でのモデル初期化テスト"""
        # モデルの読み込みをモック化
            with patch('src.clip_model.VisionTextDualEncoderModel.from_pretrained') as mock_model, \
             patch('src.clip_model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.clip_model.AutoImageProcessor.from_pretrained') as mock_processor:
            
            # モックの設定
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            mock_processor.return_value = MagicMock()
            
            # モデルの初期化
            model = CLIPModel()
            
            # アサーション
            assert model.model_name == 'rinna/japanese-clip-vit-b-16'
            assert model.device in ['cpu', 'cuda']
            mock_model.assert_called_once_with('rinna/japanese-clip-vit-b-16', trust_remote_code=True)
            mock_tokenizer.assert_called_once_with('rinna/japanese-clip-vit-b-16', trust_remote_code=True)
            mock_processor.assert_called_once_with('rinna/japanese-clip-vit-b-16', trust_remote_code=True)
    
    def test_model_initialization_custom_model(self):
        """カスタムモデル名での初期化テスト"""
        custom_model_name = 'custom/model-name'
        
            with patch('src.clip_model.VisionTextDualEncoderModel.from_pretrained') as mock_model, \
             patch('src.clip_model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.clip_model.AutoImageProcessor.from_pretrained') as mock_processor:
            
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            mock_processor.return_value = MagicMock()
            
            model = CLIPModel(model_name=custom_model_name)
            
            assert model.model_name == custom_model_name
            mock_model.assert_called_once_with(custom_model_name, trust_remote_code=True)
    
    def test_device_setting_cpu(self):
        """CPUデバイス設定のテスト"""
            with patch('src.clip_model.VisionTextDualEncoderModel.from_pretrained') as mock_model, \
             patch('src.clip_model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.clip_model.AutoImageProcessor.from_pretrained') as mock_processor:
            
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_tokenizer.return_value = MagicMock()
            mock_processor.return_value = MagicMock()
            
            model = CLIPModel(device='cpu')
            
            assert model.device == 'cpu'
            mock_model_instance.to.assert_called_once_with('cpu')
    
    def test_device_setting_cuda(self):
        """CUDAデバイス設定のテスト（CUDA利用可能な場合）"""
            with patch('src.clip_model.VisionTextDualEncoderModel.from_pretrained') as mock_model, \
             patch('src.clip_model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.clip_model.AutoImageProcessor.from_pretrained') as mock_processor:
            
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_tokenizer.return_value = MagicMock()
            mock_processor.return_value = MagicMock()
            
            model = CLIPModel(device='cuda')
            
            assert model.device == 'cuda'
            mock_model_instance.to.assert_called_once_with('cuda')
    
    def test_environment_variable_loading(self):
        """環境変数からの設定読み込みテスト"""
        with patch.dict(os.environ, {'MODEL_NAME': 'test/model', 'DEVICE': 'cpu'}), \
            patch('src.clip_model.VisionTextDualEncoderModel.from_pretrained') as mock_model, \
             patch('src.clip_model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.clip_model.AutoImageProcessor.from_pretrained') as mock_processor:
            
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            mock_processor.return_value = MagicMock()
            
            model = CLIPModel()
            
            assert model.model_name == 'test/model'
            assert model.device == 'cpu'
    
    def test_model_loading_failure(self):
        """モデル読み込み失敗時のエラーハンドリングテスト"""
            with patch('src.clip_model.VisionTextDualEncoderModel.from_pretrained') as mock_model:
            mock_model.side_effect = Exception("Model not found")
            
            with pytest.raises(RuntimeError) as exc_info:
                CLIPModel()
            
            assert "モデルの読み込みに失敗しました" in str(exc_info.value)
    
    def test_get_model_info(self):
        """モデル情報取得のテスト"""
            with patch('src.clip_model.VisionTextDualEncoderModel.from_pretrained') as mock_model, \
             patch('src.clip_model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.clip_model.AutoImageProcessor.from_pretrained') as mock_processor:
            
            # モックモデルにパラメータを追加
            mock_model_instance = MagicMock()
            mock_param = MagicMock()
            mock_param.numel.return_value = 1000
            mock_model_instance.parameters.return_value = [mock_param, mock_param]
            mock_model.return_value = mock_model_instance
            mock_tokenizer.return_value = MagicMock()
            mock_processor.return_value = MagicMock()
            
            model = CLIPModel(device='cpu')
            info = model.get_model_info()
            
            assert 'model_name' in info
            assert 'device' in info
            assert 'parameters' in info
            assert 'cuda_available' in info
            assert info['model_name'] == 'rinna/japanese-clip-vit-b-16'
            assert info['device'] == 'cpu'
            assert info['parameters'] == 2000
    
    def test_repr(self):
        """__repr__メソッドのテスト"""
            with patch('src.clip_model.VisionTextDualEncoderModel.from_pretrained') as mock_model, \
             patch('src.clip_model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('src.clip_model.AutoImageProcessor.from_pretrained') as mock_processor:
            
            mock_model.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            mock_processor.return_value = MagicMock()
            
            model = CLIPModel(device='cpu')
            repr_str = repr(model)
            
            assert 'CLIPModel' in repr_str
            assert 'rinna/japanese-clip-vit-b-16' in repr_str
            assert 'cpu' in repr_str
