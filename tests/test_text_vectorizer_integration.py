import os
import unittest

from utils.text_vectorizer import text_to_vector


@unittest.skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS") == "1",
    "Set RUN_INTEGRATION_TESTS=1 to run integration tests.",
)
class TextVectorizerIntegrationTests(unittest.TestCase):
    def test_text_to_vector_real_model(self):
        text = "한국어 문장을 실제 모델로 임베딩합니다."
        vector = text_to_vector(text)

        print("Real model vector:", vector)
        print("Real model vector length:", len(vector))
        print("Real model vector head:", vector[:5])
        self.assertTrue(len(vector) > 0)
        self.assertTrue(all(isinstance(v, float) for v in vector))


if __name__ == "__main__":
    unittest.main()
