from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import shutil
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.github.ai/inference"

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10000,
    chunk_overlap = 100
    )
    chunks = text_splitter.split_text(text)
    return chunks

def mistralai_embedding_model():
    """
    Returns a LangChain-compatible MistralAIEmbeddings model for generating 1024-dim vectors.
    Does not perform any embedding calls.

    Returns:
        MistralAIEmbeddings: Ready-to-use embedding model instance.
    """
    api_key = os.environ["MISTRAL_API_KEY"]
    embed_model = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key
    )
    return embed_model

def mistralai_llm_model(model_name: str = "mistral-small-2503", temperature: float = 0.1):
    """
    Returns a LangChain-compatible Mistral LLM instance.

    Parameters:
        model_name (str): Mistral model name (default: 'mistral-small-3.1')
        temperature (float): Sampling temperature
        api_key (str): Mistral API key (optional if set via env)

    Returns:
        ChatMistralAI: LangChain chat model instance
    """
    return ChatMistralAI(
            base_url = endpoint,
            api_key= github_token,
            model=model_name,
            temperature=temperature
            )

def embedding_model(model='text-embedding-3-small', dimensions=1536):
    embed_model = OpenAIEmbeddings(
            model=model,
            dimensions=dimensions,
            base_url = endpoint,
            api_key= github_token)
    return embed_model

def llm_model(model_name = "openai/gpt-4.1-nano", temperature = 0.1):
    model = ChatOpenAI(
        base_url = endpoint,
        api_key= github_token,
        model=model_name,
        temperature=temperature
        )
    return model

def vector_store_using_chunks(text_chunks, embedding_model):
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_selected_resume")

def create_and_save_faiss_store_from_documents(documents: list, embedding_model, save_path: str) -> FAISS:
    """
    Creates a FAISS vector store from LangChain Documents and saves it locally.
    """
    vector_store = FAISS.from_documents(documents, embedding=embedding_model)
    vector_store.save_local(save_path)
    return vector_store

def get_vector_embeddings(text, embedding_llm):
    """
    This functions retuns the vector embeddings for the given text.
    """
    vector = embedding_llm.embed_query(text)
    return vector

def delete_faiss_store(store_path: str, verbose: bool = True) -> None:
    """
    Deletes the FAISS store directory and its contents.

    Args:
        store_path (str): Path to the FAISS store directory.
        verbose (bool): Whether to print status messages.
    """
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
        if verbose:
            print(f"FAISS store at '{store_path}' has been deleted.")
    else:
        if verbose:
            print(f"No FAISS store found at '{store_path}'. Nothing to delete.")