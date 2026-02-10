
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # 一時ディレクトリの作成
        self.test_dir = Path(tempfile.mkdtemp())
        self.vector_dir = self.test_dir / "vectors"
        self.metadata_dir = self.test_dir / "metadata"
        self.store = VectorStore(self.vector_dir, self.metadata_dir)

    def tearDown(self):
        # 一時ディレクトリの削除
        shutil.rmtree(self.test_dir)

    def test_init_creates_directories(self):
        """初期化時にディレクトリが作成されることを確認"""
        self.assertTrue(self.vector_dir.exists())
        self.assertTrue(self.metadata_dir.exists())

    def test_save_and_load(self):
        """保存と読み込みが正しく行われることを確認"""
        vectors = np.random.rand(5, 512).astype(np.float32)
        metadata = [{"id": i, "text": f"item_{i}"} for i in range(5)]

        self.store.save(vectors, metadata)

        loaded_vectors, loaded_metadata = self.store.load()

        self.assertIsNotNone(loaded_vectors)
        self.assertIsNotNone(loaded_metadata)
        
        # ベクトルの一致確認
        np.testing.assert_array_equal(vectors, loaded_vectors)
        
        # メタデータの一致確認
        self.assertEqual(metadata, loaded_metadata)

    def test_save_mismatch_counts(self):
        """ベクトルとメタデータの数が一致しない場合にエラーが発生することを確認"""
        vectors = np.random.rand(5, 512)
        metadata = [{"id": i} for i in range(4)] # 1つ少ない

        with self.assertRaises(ValueError):
            self.store.save(vectors, metadata)

    def test_load_non_existent(self):
        """ファイルが存在しない場合はNoneが返されることを確認"""
        vectors, metadata = self.store.load()
        self.assertIsNone(vectors)
        self.assertIsNone(metadata)

    def test_load_inconsistency(self):
        """読み込んだデータの数が一致しない場合にエラー（または警告）になるか確認
        
        ここでは手動で不整合なファイルを作成してテストする
        """
        # 正しいデータで一度保存
        vectors = np.random.rand(5, 512).astype(np.float32)
        metadata = [{"id": i} for i in range(5)]
        self.store.save(vectors, metadata)

        # メタデータだけ書き換えて少なくする
        item_metadata_path = self.metadata_dir / "metadata.json"
        short_metadata = [{"id": i} for i in range(4)]
        with open(item_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(short_metadata, f)

        # 読み込み時にエラーになることを期待
        with self.assertRaises(IOError):
            self.store.load()

if __name__ == '__main__':
    unittest.main()
