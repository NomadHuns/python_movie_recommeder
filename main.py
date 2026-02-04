from fastapi import FastAPI, Query
from utils.chroma_utils import query_movies, save_movies_to_chroma
from contextlib import asynccontextmanager
import os

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
    results = query_movies(
        query_text=query, 
        n_results=5, 
        collection_name=collection_name, 
        persist_directory=persist_directory
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
