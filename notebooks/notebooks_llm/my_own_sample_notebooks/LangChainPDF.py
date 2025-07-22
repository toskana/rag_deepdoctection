# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LangChain PDF Frage-Antwort Beispiel mit SentenceTransformerEmbeddings
#
# In diesem Notebook laden wir eine PDF, erzeugen Embeddings mit `SentenceTransformerEmbeddings` und stellen Fragen zur PDF mit einem einfachen lokalen LLM Wrapper.

# %%
# Installiere nötige Pakete (einmalig)
# !pip install langchain pypdf sentence-transformers faiss-cpu requests

# %%
# Imports
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests


# %%
# Einfacher Wrapper für dein lokales LLM im Docker-Container (llm)
class LocalLlamaLLM(LLM):
    def __init__(self, endpoint: str = "http://llm:5000/completion"):
        self.endpoint = endpoint

    @property
    def _llm_type(self) -> str:
        return "local_llama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        json_data = {"prompt": prompt, "n_predict": 200}
        response = requests.post(self.endpoint, json=json_data)
        if response.status_code == 200:
            return response.json()["content"]
        else:
            raise RuntimeError(f"LLM Anfrage fehlgeschlagen: {response.status_code} - {response.text}")


# %%

# %%
# Lade PDF und teile in Dokumente
pdf_path = "example.pdf"  # Pfad zur PDF-Datei
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Embeddings mit SentenceTransformer
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Erstelle FAISS Vektor-Datenbank
vectorstore = FAISS.from_documents(documents, embeddings)


# %%
# Setup Retrieval-QA Chain mit LocalLlamaLLM
llm = LocalLlamaLLM()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())


# %%
# Beispiel-Frage an die PDF
frage = "Was ist das Thema der PDF?"
antwort = qa_chain.run(frage)
print("Frage:", frage)
print("Antwort:", antwort)

# %%
