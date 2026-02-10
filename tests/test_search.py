import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.search import SearchEngine
from src.vector_store import VectorStore

class TestSearchEngine:
    @pytest.fixture
    def mock_vector_store(self):
        """VectorStoreのモックを作成"""
        store = MagicMock(spec=VectorStore)
        return store

    @pytest.fixture
    def sample_data(self):
        """テスト用のベクトルとメタデータを作成"""
        # 3つのベクトル (2次元)
        # v1: [1, 0]
        # v2: [0, 1]
        # v3: [0.707, 0.707] (45度)
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.70710678, 0.70710678]
        ], dtype=np.float32)
        
        metadata = [
            {'path': 'img1.jpg', 'id': 1},
            {'path': 'img2.jpg', 'id': 2},
            {'path': 'img3.jpg', 'id': 3}
        ]
        return vectors, metadata

    def test_init_loads_data(self, mock_vector_store, sample_data):
        """初期化時にデータをロードすることを確認"""
        vectors, metadata = sample_data
        mock_vector_store.load.return_value = (vectors, metadata)
        
        engine = SearchEngine(mock_vector_store)
        
        mock_vector_store.load.assert_called_once()
        np.testing.assert_array_equal(engine.vectors, vectors)
        assert engine.metadata == metadata

    def test_init_no_data(self, mock_vector_store):
        """データがない場合でも初期化できるが警告が出ることを確認（警告のテストは省略、エラーにならないこと）"""
        mock_vector_store.load.return_value = (None, None)
        
        engine = SearchEngine(mock_vector_store)
        assert engine.vectors is None
        assert engine.metadata is None

    def test_search_basic(self, mock_vector_store, sample_data):
        """基本的な検索動作の確認"""
        vectors, metadata = sample_data
        mock_vector_store.load.return_value = (vectors, metadata)
        engine = SearchEngine(mock_vector_store)
        
        # クエリ: [1, 0] -> img1と完全一致
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = engine.search(query, top_k=3)
        
        assert len(results) == 3
        # 1位は img1 (score 1.0)
        assert results[0]['image_path'] == 'img1.jpg'
        assert pytest.approx(results[0]['score'], 0.0001) == 1.0
        
        # 2位は img3 (score ~0.707)
        assert results[1]['image_path'] == 'img3.jpg'
        assert pytest.approx(results[1]['score'], 0.0001) == 0.70710678
        
        # 3位は img2 (score 0.0)
        assert results[2]['image_path'] == 'img2.jpg'
        assert pytest.approx(results[2]['score'], 0.0001) == 0.0

    def test_search_top_k(self, mock_vector_store, sample_data):
        """top_kが機能することを確認"""
        vectors, metadata = sample_data
        mock_vector_store.load.return_value = (vectors, metadata)
        engine = SearchEngine(mock_vector_store)
        
        query = np.array([1.0, 0.0])
        results = engine.search(query, top_k=1)
        
        assert len(results) == 1
        assert results[0]['image_path'] == 'img1.jpg'

    def test_search_no_data_error(self, mock_vector_store):
        """データがない状態で検索するとエラーになることを確認"""
        mock_vector_store.load.return_value = (None, None)
        engine = SearchEngine(mock_vector_store)
        
        query = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="Vector data is not loaded"):
            engine.search(query)

    def test_search_dimension_mismatch(self, mock_vector_store, sample_data):
        """次元数が合わない場合にエラーになることを確認"""
        vectors, metadata = sample_data
        mock_vector_store.load.return_value = (vectors, metadata)
        engine = SearchEngine(mock_vector_store)
        
        # 3次元ベクトルでの検索（データは2次元）
        query = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            engine.search(query)
