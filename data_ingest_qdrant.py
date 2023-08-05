import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Batch

# # # Load env keys # # #

load_dotenv()
collection_name = os.getenv("QDRANT_COLLECTION_NAME")
client = QdrantClient(
    os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY")
)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# # # # # #


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_collection():
    vectors_config = VectorParams(size=1536, distance=Distance.COSINE)

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )


def create_vectorstore():
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    return vectorstore


def get_pdf_text(docs_dir_path):
    text = ""
    pdf_docs = os.listdir(docs_dir_path)
    for pdf in pdf_docs:
        pdf_reader = PdfReader(f"{docs_dir_path}/{pdf}")
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def load_documents(directory_path):
    folders = os.listdir(directory_path)
    for folder in folders:
        raw_text = get_pdf_text(f"{directory_path}/{folder}")
        texts = get_text_chunks(raw_text)
        embedd = embeddings.embed_documents(texts, 1000)
        ids = list(range(len(texts)))
        assert len(texts) == len(embedd) and len(texts) == len(
            ids
        ), "Las dimensiones de los embeddings no coinciden"
        meta = [{"curso": folder, "raw_text": text} for text in texts]
        client.upsert(
            collection_name=collection_name,
            points=Batch(ids=ids, vectors=embedd, payloads=meta),
        )
        collection_vector_count = client.get_collection(
            collection_name=collection_name
        ).vectors_count
        print(f"Vectores en la coleccion: {collection_vector_count}")
