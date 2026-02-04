from fastapi import FastAPI, Query
from utils.chroma_utils import query_movies, save_movies_to_chroma
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import os

def extract_filters(query: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    사용자의 질문에서 필터링 조건(국가, 장르 등)을 추출합니다.
    """
    # 국가 필터링 (간단한 키워드 매칭)
    countries = {
        "한국": "한국", "대한민국": "한국", "우리나리": "한국",
        "미국": "미국", "할리우드": "미국",
        "일본": "일본",
        "영국": "영국",
        "프랑스": "프랑스",
        "중국": "중국",
        "독일": "독일",
        "홍콩": "홍콩"
    }
    
    # 필터 조건들을 담을 리스트
    where_list = []
    where_document_list = []
    
    # 국가 필터 추가
    for k, v in countries.items():
        if k in query:
            where_list.append({"country": v})
            break

    # 장르 필터링 (간단한 키워드 매칭)
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

    for k, v in genres.items():
        if k in query:
            # ChromaDB의 where 필터는 문자열에 대해 $contains를 지원하지 않으므로
            # where_document를 사용하여 문서 내용(줄거리 등)에서 키워드를 찾습니다.
            where_document_list.append({"$contains": v})
    
    # where 필터 구성
    where_filter = None
    if len(where_list) == 1:
        where_filter = where_list[0]
    elif len(where_list) > 1:
        where_filter = {"$and": where_list}
        
    # where_document 필터 구성
    where_document_filter = None
    if len(where_document_list) == 1:
        where_document_filter = where_document_list[0]
    elif len(where_document_list) > 1:
        where_document_filter = {"$and": where_document_list}
    
    return where_filter, where_document_filter

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
    where_filter, where_document_filter = extract_filters(query)
    if where_filter:
        print(f"Applying where filters: {where_filter}")
    if where_document_filter:
        print(f"Applying where_document filters: {where_document_filter}")

    results = query_movies(
        query_text=query, 
        n_results=5, 
        collection_name=collection_name, 
        persist_directory=persist_directory,
        where=where_filter,
        where_document=where_document_filter
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
