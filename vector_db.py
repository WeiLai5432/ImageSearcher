from pymilvus import MilvusClient
from utils import timer


class VectorDB:
    def __init__(self, db_file):
        self.db_file = db_file
        self.client = MilvusClient(db_file)

    def create_collection(
        self, collection_name, vector_field_name="vector", dimension=512
    ):
        if self.client.has_collection(collection_name=collection_name):
            # self.client.drop_collection(collection_name=collection_name)
            raise Exception("Collection already exists")
        self.client.create_collection(
            collection_name=collection_name,
            vector_field_name=vector_field_name,
            dimension=dimension,
            auto_id=True,
            enable_dynamic_field=True,
            metric_type="COSINE",
        )

    def insert_embedding(self, collection_name, embedding, file_name):
        return self.client.insert(
            collection_name=collection_name,
            data={"vector": embedding, "file_name": file_name},
        )

    @timer("search db")
    def search(self, collection_name, embedding, limit=10):
        results = self.client.search(
            collection_name=collection_name,
            data=[embedding],
            output_fields=["file_name"],
            search_params={"metrics_type": "COSINE"},
            limit=limit,
        )
        images_probs = []
        for result in results:
            for hit in result:
                images_probs.append((hit["entity"]["file_name"], hit["distance"]))
        return images_probs
