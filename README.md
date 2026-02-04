# Python Movie Recommender

## 텍스트 벡터화(임베딩)

이 프로젝트는 Hugging Face 임베딩 모델 `nlpai-lab/KURE-v1`를 사용해 텍스트를
숫자 벡터로 변환합니다.

```python
from utils.text_vectorizer import text_to_vector

vector = text_to_vector("우정에 관한 따뜻한 성장 드라마")
print(len(vector), vector[:5])
```

참고: 첫 실행 시 Hugging Face에서 모델을 다운로드하므로 시간이 걸릴 수 있습니다.

## 벡터 DB에 저장하는 명령어
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 utils/chroma_utils.py
```

가상환경에서는 아래로 명령어 실행
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd) && ./venv/bin/python3 utils/chroma_utils.py
```

## 서버 실행 명령어
```bash
uvicorn main:app --reload
```

## 영화 추천 API (`/test`)

사용자가 영화 추천 질문을 쿼리 스트링으로 보내면, 벡터 검색을 통해 가장 유사한 영화 5개를 JSON 형태로 반환합니다.

### 요청 예시
```bash
curl "http://localhost:8000/test?query=슬픈영화추천해줘"
```

### 응답 예시
```json
[
  {
    "title": "영화 제목",
    "url": "http://...",
    "genre": "드라마",
    "country": "한국",
    "year": "2024",
    "synopsis": "영화 줄거리 요약...",
    "similarity_score": 0.8542
  },
  ...
]
```
