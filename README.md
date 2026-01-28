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
