
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# srcモジュールへのパスを追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts import build_index

class TestBuildIndexScript(unittest.TestCase):
    def setUp(self):
        self.image_dir = Path("tests/temp_images")
        self.vector_dir = Path("tests/temp_vectors")
        self.metadata_dir = Path("tests/temp_metadata")
        
        # ディレクトリ作成
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # ダミー画像作成
        with open(self.image_dir / "test.jpg", "w") as f:
            f.write("dummy")
            
    def tearDown(self):
        # ディレクトリ削除
        if self.image_dir.exists():
            shutil.rmtree(self.image_dir)
        if self.vector_dir.exists():
            shutil.rmtree(self.vector_dir)
        if self.metadata_dir.exists():
            shutil.rmtree(self.metadata_dir)

    @patch('scripts.build_index.CLIPModel')
    @patch('scripts.build_index.ImageEncoder')
    @patch('scripts.build_index.VectorStore')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main(self, mock_args, mock_store, mock_encoder, mock_clip):
        # モックの設定
        mock_args_instance = MagicMock()
        mock_args_instance.image_dir = str(self.image_dir)
        mock_args_instance.vector_dir = str(self.vector_dir)
        mock_args_instance.metadata_dir = str(self.metadata_dir)
        mock_args_instance.batch_size = 2
        mock_args_instance.model_name = None
        mock_args_instance.device = None
        mock_args.return_value = mock_args_instance
        
        # ImageEncoderのモック
        mock_encoder_instance = mock_encoder.return_value
        mock_encoder_instance.encode_images_batch.return_value = [[0.1] * 512]
        
        # VectorStoreのモック
        mock_store_instance = mock_store.return_value
        
        # メイン関数の実行
        build_index.main()
        
        # 検証
        mock_clip.assert_called_once()
        mock_encoder.assert_called_once()
        mock_store.assert_called_once_with(self.vector_dir, self.metadata_dir)
        mock_encoder_instance.encode_images_batch.assert_called_once()
        mock_store_instance.save.assert_called_once()

if __name__ == '__main__':
    unittest.main()
