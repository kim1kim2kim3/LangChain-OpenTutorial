"""
Chroma 검색기에서 필터를 사용하는 예시
"""

from utils.chroma import ChromaDocumentMangager
from langchain_openai import OpenAIEmbeddings
import chromadb

# Chroma 클라이언트와 임베딩 모델 초기화
client = chromadb.Client()
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# 문서 관리자 인스턴스 생성
crud_manager = ChromaDocumentMangager(client=client, embedding=embedding)

def example1_basic_search():
    """필터 없이 기본 검색"""
    ret = crud_manager.as_retriever(
        search_fn=crud_manager.search,
        search_kwargs={"k": 3}  # 필터 없이 상위 3개 결과 반환
    )
    print("예제 1: 필터 없는 기본 검색")
    print(ret.invoke("Which asteroid did the little prince come from?"))
    print("\n" + "="*50 + "\n")

def example2_title_filter():
    """특정 제목 필터를 사용한 검색"""
    ret = crud_manager.as_retriever(
        search_fn=crud_manager.search,
        search_kwargs={
            "k": 2,
            "where": {"title": "Chapter 4"}  # 4장만 검색하도록 필터 적용
        }
    )
    print("예제 2: 제목 필터 사용 (4장)")
    print(ret.invoke("Which asteroid did the little prince come from?"))
    print("\n" + "="*50 + "\n")

def example3_multiple_filters():
    """$in 연산자를 사용한 다중 필터 검색"""
    ret = crud_manager.as_retriever(
        search_fn=crud_manager.search,
        search_kwargs={
            "k": 2,
            "where": {
                "title": {"$in": ["Chapter 21", "Chapter 24", "Chapter 25"]}  # 특정 장에서만 검색
            }
        }
    )
    print("예제 3: 여우의 비밀이 등장하는 특정 장 검색")
    print(ret.invoke("What is essential is invisible to the eye?"))

if __name__ == "__main__":
    # 모든 예제 실행
    example1_basic_search()
    example2_title_filter()
    example3_multiple_filters()