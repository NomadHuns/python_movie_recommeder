import chromadb
from typing import List, Dict, Any
from utils.csv_vectorizer import vectorize_csv
from pathlib import Path

def save_movies_to_chroma(
    csv_path: str,
    collection_name: str = "movies",
    persist_directory: str = "./chroma_db",
    text_columns: List[str] = ["title", "synopsis"],
):
    """
    CSV 파일을 읽어 벡터화한 후 ChromaDB에 저장합니다.
    """
    # 1. CSV 데이터 벡터화
    print(f"Vectorizing data from {csv_path}...")
    # 주의: 데이터 양이 많으면 시간이 오래 걸릴 수 있습니다.
    results = vectorize_csv(csv_path, text_columns=text_columns)
    
    if not results:
        print("No data found to vectorize.")
        return

    # 2. ChromaDB 클라이언트 설정 (Persistent Storage)
    client = chromadb.PersistentClient(path=persist_directory)
    
    # 3. 컬렉션 생성 또는 가져오기
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    
    collection = client.create_collection(name=collection_name)

    # 4. 데이터 준비
    ids = []
    vectors = []
    metadatas = []
    documents = []

    for i, item in enumerate(results):
        ids.append(f"movie_{i}")
        vectors.append(item["vector"])
        
        # 메타데이터 (row 데이터에서 ChromaDB 호환 타입만 추출)
        raw_row = item["row"]
        metadata = {}
        for k, v in raw_row.items():
            if isinstance(v, (str, int, float, bool)):
                metadata[k] = v
            else:
                metadata[k] = str(v)
        metadatas.append(metadata)
        
        documents.append(item["text"])

    # 5. ChromaDB에 저장 (배치 처리)
    print(f"Saving {len(ids)} items to ChromaDB collection '{collection_name}'...")
    
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=vectors[i:end_idx],
            metadatas=metadatas[i:end_idx],
            documents=documents[i:end_idx]
        )
        print(f"Inserted {end_idx}/{len(ids)} items...")

    print("Successfully saved to ChromaDB.")

def query_movies(query_text: str, n_results: int = 5, collection_name: str = "movies", persist_directory: str = "./chroma_db"):
    """
    ChromaDB에서 쿼리와 유사한 영화를 검색합니다.
    """
    from utils.text_vectorizer import text_to_vector
    
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)
    
    query_vector = text_to_vector(query_text)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    return results

if __name__ == "__main__":
    # 간단한 테스트 스크립트 (첫 10개 행만 벡터화하여 저장 시연용으로 수정해서 사용 가능)
    # 현재는 전체 데이터를 처리하도록 되어 있음.
    import sys
    
    current_dir = Path(__file__).parent.parent
    csv_file = current_dir / "dataset" / "movie_data.csv"
    
    if csv_file.exists():
        # 실제 실행 시에는 시간이 걸릴 수 있음을 안내
        print("Note: Processing the entire dataset may take several minutes.")
        save_movies_to_chroma(str(csv_file))
    else:
        print(f"File not found: {csv_file}")
