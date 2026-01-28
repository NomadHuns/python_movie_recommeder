import csv
import os
import tempfile
import unittest
from pathlib import Path

from utils.csv_vectorizer import save_vectors_to_csv, save_vectors_to_json, vectorize_csv


@unittest.skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS") == "1",
    "Set RUN_INTEGRATION_TESTS=1 to run integration tests.",
)
class CsvVectorizerIntegrationTests(unittest.TestCase):
    def test_vectorize_and_save_real_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "sample.csv"
            output_csv = temp_path / "vectors.csv"
            output_json = temp_path / "vectors.json"

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["title", "url", "synopsis", "genre", "country", "year"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "title": "작은 마을의 이야기",
                        "url": "https://example.com/movie/1",
                        "synopsis": "작은 마을에서 벌어지는 우정에 관한 이야기.",
                        "genre": "드라마",
                        "country": "한국",
                        "year": "2020",
                    }
                )
                writer.writerow(
                    {
                        "title": "미스터리 사건",
                        "url": "https://example.com/movie/2",
                        "synopsis": "형사가 의문의 사건을 수사한다.",
                        "genre": "미스터리",
                        "country": "영국",
                        "year": "2018",
                    }
                )

            results = vectorize_csv(str(csv_path))
            self.assertEqual(len(results), 2)
            print("Sample vector length:", len(results[0]["vector"]))
            print("Sample vector head:", results[0]["vector"][:5])

            save_vectors_to_csv(results, str(output_csv))
            save_vectors_to_json(results, str(output_json))

            self.assertTrue(output_csv.exists())
            self.assertTrue(output_json.exists())

            with output_csv.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertIn("title", reader.fieldnames or [])
                self.assertIn("v0", reader.fieldnames or [])
                rows = list(reader)
                self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
