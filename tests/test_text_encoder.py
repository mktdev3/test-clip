import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
from src.clip_model import CLIPModel
from src.text_encoder import TextEncoder

class TestTextEncoder(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock(spec=CLIPModel)
        self.mock_model.device = 'cpu'
        
        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        def tokenizer_side_effect(text, **kwargs):
            if isinstance(text, list):
                batch_size = len(text)
            else:
                batch_size = 1
            return {
                'input_ids': torch.ones((batch_size, 10)),
                'attention_mask': torch.ones((batch_size, 10))
            }
        self.mock_tokenizer.side_effect = tokenizer_side_effect
        self.mock_model.tokenizer = self.mock_tokenizer
        
        # Mock model
        self.mock_transformer = MagicMock()
        def get_text_features(**kwargs):
            input_ids = kwargs.get('input_ids')
            batch_size = input_ids.shape[0]
            return torch.rand((batch_size, 512))
        self.mock_transformer.get_text_features = MagicMock(side_effect=get_text_features)
        self.mock_model.model = self.mock_transformer

    def test_initialization(self):
        encoder = TextEncoder(self.mock_model)
        self.assertEqual(encoder.clip_model, self.mock_model)

    def test_initialization_error(self):
        with self.assertRaises(ValueError):
            TextEncoder("invalid_model")

    def test_normalize_text(self):
        encoder = TextEncoder(self.mock_model)
        text = " Ｈｅｌｌｏ  "
        self.assertEqual(encoder._normalize_text(text), "Hello")
        text_jp = "　こんにちは　"
        self.assertEqual(encoder._normalize_text(text_jp), "こんにちは")

    def test_normalize_empty_text(self):
        encoder = TextEncoder(self.mock_model)
        with self.assertRaises(ValueError):
            encoder._normalize_text("")
        with self.assertRaises(ValueError):
            encoder._normalize_text("   ")

    def test_encode_single_text(self):
        encoder = TextEncoder(self.mock_model)
        text = "cat"
        vector = encoder.encode_text(text)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (512,))

    def test_encode_batch_texts(self):
        encoder = TextEncoder(self.mock_model)
        texts = ["cat", "dog", "bird"]
        vectors = encoder.encode_text(texts)
        self.assertIsInstance(vectors, np.ndarray)
        self.assertEqual(vectors.shape, (3, 512))

    def test_encode_batch_processing(self):
        encoder = TextEncoder(self.mock_model)
        texts = ["text"] * 10
        vectors = encoder.encode_text(texts, batch_size=3)
        self.assertEqual(vectors.shape, (10, 512))
        # tokenizer call count check
        self.assertGreaterEqual(self.mock_model.tokenizer.call_count, 4)

    def test_encode_invalid_input(self):
        encoder = TextEncoder(self.mock_model)
        with self.assertRaises(ValueError):
            encoder.encode_text(123)
