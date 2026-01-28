import math
import types
import unittest
from unittest import mock

import torch

from utils import text_vectorizer


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None, truncation=None):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class _FakeModel:
    def __call__(self, **kwargs):
        # Shape: (batch, seq, hidden)
        hidden = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]]
        )
        return types.SimpleNamespace(last_hidden_state=hidden)


class TextVectorizerTests(unittest.TestCase):
    def test_text_to_vector_returns_normalized_list(self):
        fake_tokenizer = _FakeTokenizer()
        fake_model = _FakeModel()

        with mock.patch.object(text_vectorizer, "_get_model", return_value=(fake_tokenizer, fake_model)):
            vector = text_vectorizer.text_to_vector("test")

        print("Vector:", vector)
        self.assertEqual(len(vector), 4)
        self.assertTrue(all(isinstance(v, float) for v in vector))
        norm = math.sqrt(sum(v * v for v in vector))
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_text_to_vector_with_korean_text(self):
        fake_tokenizer = _FakeTokenizer()
        fake_model = _FakeModel()
        korean_text = "한국어 문장이 임베딩으로 잘 변환되는지 확인합니다."

        with mock.patch.object(text_vectorizer, "_get_model", return_value=(fake_tokenizer, fake_model)):
            vector = text_vectorizer.text_to_vector(korean_text)

        print("Korean vector:", vector)
        self.assertEqual(len(vector), 4)
        self.assertTrue(all(isinstance(v, float) for v in vector))

if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TextVectorizerTests)
    )
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
