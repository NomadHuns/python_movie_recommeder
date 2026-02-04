from fastapi import FastAPI, Query
from utils.chroma_utils import query_movies, save_movies_to_chroma
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import os

def extract_filters(query: str) -> Dict[str, Any]:
    """
    사용자의 질문에서 필터링 조건(국가, 장르 등)을 추출합니다.
    """
    where = {}
    
    # 국가 필터링 (간단한 키워드 매칭)
    countries = {
        "한국": "한국", "대한민국": "한국",
        "미국": "미국",
        "일본": "일본",
        "영국": "영국",
        "프랑스": "프랑스",
        "중국": "중국",
        "독일": "독일",
        "홍콩": "홍콩"
    }
    
    for k, v in countries.items():
        if k in query:
            where["country"] = v
            break

    # 장르 필터링 (간단한 키워드 매칭)
    # 데이터셋의 장르가 "범죄/스릴러" 형태이므로 $contains 연산자를 사용하는 것이 좋지만,
    # ChromaDB의 기본 where 필터는 일치 여부를 확인하므로 여기서는 $contains 스타일의 로직을 고려해야 합니다.
    # 하지만 현재 collection.add 시 metadata에 문자열로 저장했으므로, 
    # ChromaDB 버전에 따라 $contains 지원 여부가 다를 수 있습니다.
    # 안전하게 여기서는 키워드가 포함된 경우 해당 키워드를 필터로 사용하되, 
    # 검색 성능 향상을 위해 '공포', '액션' 등의 주요 키워드를 체크합니다.
    genres = {
        "공포": "공포", "호러": "공포",
        "액션": "액션",
        "코미디": "코미디",
        "SF": "SF",
        "로맨스": "로맨스", "멜로": "로맨스",
        "스릴러": "스릴러",
        "애니메이션": "애니메이션",
        "다큐멘터리": "다큐멘터리",
        "드라마": "드라마",
        "판타지": "판타지",
        "범죄": "범죄",
        "미스터리": "미스터리"
    }

    genre_filters = []
    for k, v in genres.items():
        if k in query:
            genre_filters.append(v)
    
    # 필터가 여러 개일 경우 ChromaDB의 $and 등을 사용할 수 있으나, 
    # 여기서는 가장 명확한 국가와 장르 하나씩만 우선 적용해 봅니다.
    # 장르의 경우 '장르1/장르2' 형태이므로 정규식이나 $contains가 필요할 수 있습니다.
    # 일단 'country' 필터만이라도 확실히 적용합니다.
    
    # 장르 필터는 데이터 형태 때문에 일치(equal)로 찾기 어려우므로(예: '코미디/공포'),
    # ChromaDB 0.4.x 이상에서 지원하는 $contains를 사용하거나, 
    # 여기서는 우선 확실한 'country'만 적용하고 장르는 벡터 검색에 맡기는 전략을 취합니다.
    # (사용자가 '한국'을 명시했을 때 효과가 가장 큼)
    
    if "country" in where:
        # 국가 필터만 우선적으로 적용합니다.
        # 장르의 경우 "공포/코미디" 같이 여러 개가 섞여 있어 일치($eq)로 필터링하면 
        # 결과가 너무 적게 나올 수 있으므로, 국가만 필터링하고 장르는 벡터 검색 결과에 맡깁니다.
        pass 
    elif genre_filters:
        # 국가가 없고 장르만 있는 경우에만 장르 필터링을 시도합니다.
        where["genre"] = genre_filters[0]

    return where if where else None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 데이터베이스 확인 및 초기화
    persist_directory = "./chroma_db"
    collection_name = "movies"
    csv_path = "./dataset/movie_data.csv"
    
    import chromadb
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Collection '{collection_name}' not found. Initializing...")
        if os.path.exists(csv_path):
            save_movies_to_chroma(
                csv_path=csv_path,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        else:
            print(f"Warning: CSV file not found at {csv_path}. Collection cannot be initialized.")
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/test")
async def get_recommendations(
    query: str = Query(..., description="영화 추천 질문"),
    collection_name: str = "movies",
    persist_directory: str = "./chroma_db"
):
    """
    쿼리 스트링 query에 영화 추천 질문이 들어오면 유사한 5개의 영화 데이터를 반환합니다.
    """
    # 필터 추출
    where_filter = extract_filters(query)
    if where_filter:
        print(f"Applying filters: {where_filter}")

    results = query_movies(
        query_text=query, 
        n_results=5, 
        collection_name=collection_name, 
        persist_directory=persist_directory,
        where=where_filter
    )
    
    # 결과 가공: 리스트 형태로 변환
    movie_list = []
    if results and "metadatas" in results and results["metadatas"]:
        # ChromaDB query 결과는 리스트의 리스트 형태이므로 [0] 접근
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [None] * len(metadatas)
        
        for meta, dist in zip(metadatas, distances):
            movie_item = {
                "title": meta.get("title"),
                "url": meta.get("url"),
                "genre": meta.get("genre"),
                "country": meta.get("country"),
                "year": meta.get("year"),
                "synopsis": meta.get("synopsis"),
                "similarity_score": 1 - dist if dist is not None else None
            }
            movie_list.append(movie_item)
            
    return movie_list
