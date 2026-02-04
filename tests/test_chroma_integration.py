import unittest
from pathlib import Path
import csv
from utils.chroma_utils import save_movies_to_chroma, query_movies
import shutil
import os

class TestChromaUtils(unittest.TestCase):
    def setUp(self):
        self.test_csv = "test_movies.csv"
        self.test_db = "./test_chroma_db"
        
        # 테스트용 임시 CSV 생성
        with open(self.test_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["title", "url", "synopsis", "genre", "country", "year"])
            writer.writerow(["영화1", "http://test1.com", "슬픈 영화입니다.", "드라마", "한국", "2024"])
            writer.writerow(["영화2", "http://test2.com", "재미있는 액션 영화!", "액션", "미국", "2023"])

    def tearDown(self):
        # 테스트 파일 및 DB 삭제
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists(self.test_db):
            shutil.rmtree(self.test_db)

    def test_save_and_query(self):
        # 1. 저장 테스트
        save_movies_to_chroma(
            csv_path=self.test_csv,
            collection_name="test_collection",
            persist_directory=self.test_db
        )
        
        # 2. 조회 테스트
        results = query_movies(
            query_text="슬픈 이야기",
            n_results=1,
            collection_name="test_collection",
            persist_directory=self.test_db
        )
        
        self.assertEqual(len(results["ids"][0]), 1)
        self.assertEqual(results["metadatas"][0][0]["title"], "영화1")

if __name__ == "__main__":
    unittest.main()
