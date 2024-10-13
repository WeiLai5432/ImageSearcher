import os
from tqdm import tqdm
from clip_model import CLIPModel
from vector_db import VectorDB
from utils import timer


model_id = "ViT-B/32"
photos_folder = "./photos"
db_file = "./milvus.db"
collection_name = "image_embeddings"

image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

vector_db = VectorDB(db_file=db_file)
clip_model = CLIPModel(model_id)


def list_images(source_dir):
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                source_path = os.path.join(root, file)
                image_files.append(source_path)
                # yield source_path

    return image_files


@timer()
def build_db(folder):
    vector_db.create_collection(collection_name)
    print("Building DB with photos...")

    for image_path in tqdm(list_images(folder)):
        img_embedding = clip_model.encode_image(image_path)
        vector_db.insert_embedding(collection_name, img_embedding, image_path)


if __name__ == "__main__":
    build_db(photos_folder)
