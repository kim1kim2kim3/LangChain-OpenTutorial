from utils.base import DocumentManager
from utils.base import Iterable, Any, Optional, List, Dict
from chromadb.api import ClientAPI  # 클라이언트 타입
from chromadb.utils import embedding_functions
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding: Any):
        self.embedding = embedding

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedding.embed_documents(input)


class ChromaDocumentMangager(DocumentManager):
    def __init__(
        self, client: ClientAPI, embedding: Optional[Any] = None, **kwargs
    ) -> None:
        """
        ### kwargs
            - name[str] : 컬렉션 이름(고유값)
            - configuration[CollectionConfiguration 또는 None] : 컬렉션별 설정
            - data_loader[DataLoadable 또는 None] : 데이터 로더 선택
            - embedding_function[Callable 또는 None] : 임베딩 함수, 기본값 -> `all-MiniLM-L6-v2`
            - metadata[Dict 또는 None] :
                - hnsw:space[str 또는 None] : l2(기본값, 제곱 L2 노름), ip(내적), cosine(코사인 거리)
                - category[str 또는 None] : 컬렉션 분류
                - created_by[str 또는 None] : 컬렉션 생성자
                - description[str 또는 None] : 컬렉션 설명
                - version[int 또는 None] : 컬렉션 버전
        """
        if "name" not in kwargs:
            kwargs["name"] = "chroma_test"
        self.client = client  # Chroma Python SDK 클라이언트

        # hnsw:space의 cosine-distance 수정 `ChromaDocumentMangager` v0.0.1
        if "metadata" in kwargs:
            if "hnsw:space" not in kwargs["metadata"]:
                kwargs["metadata"]["hnsw:space"] = "cosine"
        else:
            kwargs["metadata"] = dict({"hnsw:space": "cosine"})

        # 컬렉션 생성
        self.collection = client.get_or_create_collection(
            name=kwargs["name"],
            configuration=kwargs.get("configuration", None),
            data_loader=kwargs.get("data_loader", None),
            embedding_function=(
                embedding_functions.DefaultEmbeddingFunction()
                if embedding is None
                else CustomEmbeddingFunction(embedding)
            ),
            metadata=kwargs.get("metadata", None),
        )

        # 임베딩 객체
        self.embedding = embedding

    def upsert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        texts: 문서 또는 텍스트
        metadatas: 메타데이터
        ids: 고유 ID, 값이 None이면 자동으로 생성하여 삽입
        """

        if ids is None:  # ids가 None인 경우
            ids = [str(uuid4()) for _ in range(len(texts))]

        self.collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=texts,
        )

    def upsert_parallel(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        workers: int = 10,
        **kwargs: Any,
    ) -> None:
        # 배치 생성
        batches = []
        total = len(texts)

        # 배치를 만드는 것보다 여러 번 `upsert`를 수행하는 것이 더 빠를 수도 있음
        batches = [
            (
                texts[i : i + batch_size],
                metadatas[i : i + batch_size] if metadatas else None,
                ids[i : i + batch_size] if ids else None,
            )
            for i in range(0, total, batch_size)
        ]
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(lambda batch: self.upsert(*batch, **kwargs), batches)

    def search(self, query: str, k: int = 10, **kwargs: Any) -> List[Document]:
        """
        기본 스코어링 : 코사인 유사도
        """
        # 쿼리 임베딩
        if self.embedding is None:
            query_embed = embedding_functions.DefaultEmbeddingFunction()([query])
        else:
            query_embed = self.embedding.embed_documents([query])

        # 문서 조건 설정
        where_condition = kwargs["where"] if kwargs and "where" in kwargs else None

        where_document_condition = (
            kwargs["where_document"] if kwargs and "where_document" in kwargs else None
        )

        result = self.collection.query(
            query_embeddings=query_embed,
            n_results=k,
            where=where_condition,
            where_document=where_document_condition,
        )
        # 코사인 유사도 계산
        # Cosine Similarity = 1 - Cosine Distance
        result["distances"] = [
            list(map(lambda x: round(1 - x, 2), result["distances"][0]))
        ]

        # Langchain 문서 형식으로 변환
        return [
            Document(
                page_content=document,
                metadata={"id": id, "score": distance, **metadata},
            )
            for document, id, distance, metadata in zip(
                result["documents"][0],
                result["ids"][0],
                result["distances"][0],
                result["metadatas"][0],
            )
        ]

    def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            - ids: 삭제할 임베딩의 ID 목록.
            - filters: 삭제할 데이터를 필터링하기 위한 딕셔너리.
                - where: 조건 필터링을 위한 딕셔너리.
                - where_document: 문서 기반 필터링을 위한 딕셔너리.
        """

        try:
            # 경우 1: ids와 filters가 모두 없으면 전체 삭제
            if not ids and not filters:
                all_data = self.collection.get(include=[])
                if "ids" in all_data:
                    ids = all_data["ids"]
                    print(f"{len(ids)} data deleted")
                else:
                    return

            # 경우 2: ids만 제공된 경우 -> 해당 ID 삭제
            elif ids and not filters:
                self.collection.delete(ids=ids)
                print(f"{len(ids)} data deleted")
                return

            # 경우 3: filters만 제공된 경우 -> 필터 조건으로 삭제
            elif not ids and filters:
                where_condition = filters.get("where")
                where_document_condition = filters.get("where_document")

                # 필터에 따라 삭제할 id 조회
                filtered_data = self.collection.get(
                    include=[],
                    where=where_condition,
                    where_document=where_document_condition,
                )

                # id가 정상적으로 조회되었는지 확인
                if "ids" in filtered_data:
                    ids = filtered_data["ids"]
                    if ids:
                        self.collection.delete(ids=ids)
                        print(f"{len(ids)} data deleted")
                    else:
                        print("No matching data found for filters, nothing to delete.")
                else:
                    print("No data found with the given filters.")
                return

            # 경우 4: ids와 filters가 모두 제공된 경우 -> 예외적인 시나리오
            elif ids and filters:
                where_condition = filters.get("where")
                where_document_condition = filters.get("where_document")

                # 컬렉션에서 필터링된 id 가져오기
                filtered_data = self.collection.get(
                    include=[],
                    where=where_condition,
                    where_document=where_document_condition,
                )

                # 필터링된 id 추출
                filtered_ids = filtered_data.get("ids", [])

                # 주어진 ids와 필터링된 ids의 교집합 찾기
                intersect_ids = list(set(ids) & set(filtered_ids))

                if intersect_ids:
                    self.collection.delete(ids=intersect_ids)
                    print(f"{len(intersect_ids)} data deleted")
                else:
                    print("No matching data found for the given ids and filters.")
                return

        except ValueError as e:
            print(f"Error during deletion: {str(e)}")
