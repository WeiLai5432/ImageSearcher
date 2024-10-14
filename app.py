from clip_model import CLIPModel
from vector_db import VectorDB
import time
import streamlit as st
from PIL import Image
from utils import timer
from pillow_heif import register_heif_opener

# from stqdm import stqdm

register_heif_opener()

model_id = "ViT-B/32"
db_file = "./nas.db"
collection_name = "image_embeddings"

image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".heic"]


@st.cache_resource
def initilize():
    vector_db = VectorDB(db_file=db_file)
    clip_model = CLIPModel(model_id)
    return vector_db, clip_model


vector_db, clip_model = initilize()


def search_by_text(text, limit=10):
    embedding = clip_model.encode_text(text)
    results = vector_db.search(collection_name, embedding, limit=limit)
    return results


def search_by_image(image, limit=10):
    embedding = clip_model.encode_image(image)
    results = vector_db.search(collection_name, embedding, limit=limit)

    return results


# Streamlit UI
st.title("Image Search")


@timer()
def display_results(results):
    columns = 3
    if results:
        total_results = len(results)

        progress_bar = st.progress(0)

        cols = st.columns(columns)
        # for idx, (image_path, similarity) in enumerate(stqdm(results, desc="displaying")):
        for idx, (image_path, similarity) in enumerate(results):
            progress = (idx + 1) / total_results
            progress_bar.progress(progress)
            image = Image.open(image_path)
            with cols[idx % columns]:

                if similarity > 0.9:
                    bar_color = "#4CAF50"  # Green
                elif similarity > 0.70:
                    bar_color = "#FFC107"  # Yellow
                else:
                    bar_color = "#F44336"  # Red

                st.empty().markdown(
                    f"""
                <div style="display: flex; align-items: center;">
                    <div style="width: 100%;background-color: #e0e0e0; border-radius: 5px;">
                        <div style="width: {similarity * 100}%; background-color: {bar_color}; height: 10px; border-radius: 15px;"></div>
                    </div>
                    <span>{similarity:.4f}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.image(
                    image,
                    caption=f"{image_path}",
                    use_column_width=True,
                )

        progress_bar.success("Results displayed!")

    else:
        st.write("No results found.")


search_button = None

with st.sidebar:
    st.header("Search Parameters")

    num_results = st.slider(
        "Number of results to display", min_value=1, max_value=100, value=20
    )

    query = st.text_input("Enter a description:", placeholder="Search by text")

    uploaded_image = st.file_uploader(
        "Upload an image", type=[x.split(".")[1] for x in image_extensions]
    )
    search_button = st.button("Search")
    if uploaded_image is not None:

        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)


results = None
if search_button:
    start_time = time.time()
    if query and uploaded_image:
        # Default search by image. Input text is ignored.
        st.warning("Both text and image were provided. Searching by image.")
    if uploaded_image is not None:
        if uploaded_image is not None:
            results = search_by_image(image, limit=num_results)
    elif query:
        results = search_by_text(query, limit=num_results)
    st.write(f"Search took {time.time() - start_time: .4f} seconds")

    display_results(results)
