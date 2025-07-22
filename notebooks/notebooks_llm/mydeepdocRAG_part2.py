# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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

# %%
from langchain.schema import Document

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

# %%
from langchain.vectorstores import FAISS

# %%
from langchain.chains import RetrievalQA

# %%
from langchain.prompts import PromptTemplate

# %%
from typing import Optional, List, Dict, Mapping, Any

# %%
from langchain.embeddings import HuggingFaceEmbeddings

# %%
from sentence_transformers import SentenceTransformer

# %%
import json

# %%
import pandas as pd

# %%
from io import StringIO

# %%
import requests

# %%
from langchain.llms.base import LLM


# %%
def inspect_pages_data(pages_data, max_pages=2, max_chunks=3, max_text_len=1000, max_rows=50, max_cols=10):
    print(f"Anzahl Seiten: {len(pages_data)}")
    if len(pages_data) == 0:
        print("Keine Seiten im Datenobjekt.")
        return

    for i, page in enumerate(pages_data[:max_pages]):
        print(f"\n=== Seite {i+1} ===")
        print(f"Typ: {type(page)}")
        print(f"Keys: {list(page.keys())}")
        print(f"Seiten-Nummer: {page.get('page_number')}")
        print(f"Dateiname: {page.get('file_name')}")
        print(f"Text (erste {max_text_len} Zeichen): {page.get('text', '')[:max_text_len]!r}")

        chunks = page.get("chunks", [])
        print(f"Anzahl Chunks: {len(chunks)}")
        for j, chunk in enumerate(chunks[:max_chunks]):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                bbox, text = chunk
                print(f"  Chunk {j+1}: bbox={bbox}, text={text[:max_text_len]!r}")
            else:
                print(f"  Chunk {j+1}: {str(chunk)[:max_text_len]!r}")

        tables = page.get("tables", [])
        print(f"Anzahl Tabellen: {len(tables)}")
        for k, table in enumerate(tables):
            print(f"\n  Tabelle {k+1}: Typ={type(table)} Größe={table.shape if hasattr(table, 'shape') else 'unbekannt'}")
            if hasattr(table, "head"):
                # Ausgabe der ersten max_rows Zeilen, max_cols Spalten, als Text
                print(table.iloc[:max_rows, :max_cols].to_string(index=False))
            else:
                print(str(table)[:max_text_len])


# %%
# Nur mit transformers
def chunk_document(pages: List[Dict[str, Any]], max_tokens: int = 512) -> List[Dict[str, Any]]:
    """Fügt Text und Tabellen aus Seiten zusammen und chunked sie."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # anpassbar

    chunks = []
    current_chunk = ""
    current_meta = []

    for page in pages:
        full_text = page["text"].strip()

        # Optional: Tabellen als Text anhängen
        for df in page["tables"]:
            full_text += "\n\n" + df.to_string(index=False)

        # Chunking per Token-Limit
        tokens = tokenizer.tokenize(current_chunk + full_text)
        if len(tokens) > max_tokens:
            chunks.append({
                "text": current_chunk.strip(),
                "meta": current_meta
            })
            current_chunk = full_text
            current_meta = [page]
        else:
            current_chunk += "\n\n" + full_text
            current_meta.append(page)

    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "meta": current_meta
        })

    return chunks


# %%
def build_documents_from_pages(pages_data):
    """
    Wandelt die Liste von Seiten-Dictionaries (aus DeepDoctection) 
    in eine Liste von LangChain Document-Objekten um.
    """
    documents = []
    for page in pages_data:
        # Text der Seite
        page_text = page["text"].strip()

        # Tabellen als Text anfügen (optional)
        for df in page.get("tables", []):
            page_text += "\n\n" + df.to_string(index=False)

        # Metadaten mitgeben
        metadata = {
            "page_number": page["page_number"],
            "file_name": page.get("file_name", None),
        }

        doc = Document(page_content=page_text, metadata=metadata)
        documents.append(doc)

    return documents


# %%
def import_pages_data_from_json(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages_data = []
    for page_entry in data:
        page = {
            "page_number": page_entry.get("page_number"),
            "file_name": page_entry.get("file_name"),
            "document_id": page_entry.get("document_id"),
            "image_id": page_entry.get("image_id"),
            "width": page_entry.get("width"),
            "height": page_entry.get("height"),
            "text": page_entry.get("text"),
            "tables": []
        }

        for csv_str in page_entry.get("tables", []):
            # CSV-String zurück in DataFrame konvertieren
            df = pd.read_csv(StringIO(csv_str))
            page["tables"].append(df)

        pages_data.append(page)

    return pages_data


# %%
def load_vectorstore(speicherpfad):
    # Gleicher Embedding-Model-Name wie beim Speichern
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Vectorstore laden
    vectorstore = FAISS.load_local(speicherpfad, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return vectorstore


# %%
class LocalLlamaLLM(LLM):
    endpoint: Optional[str] = None  # Optional mit Default None

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        if device == "gpu":
            self.endpoint = "http://llm-gpu:5001/completion"
        else:
            self.endpoint = "http://llm-cpu:5000/completion"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(self.endpoint, json={"prompt": prompt, "n_predict": 100})
        response.raise_for_status()
        data = response.json()
        return data.get("content", "")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"endpoint": self.endpoint}

    @property
    def _llm_type(self) -> str:
        return "local_llama"


# %%
def print_clean_result (antwort):
    clean_result = antwort["result"].replace("Answer:", "").strip()
    print(clean_result)
    return  


# %%
pages_data = import_pages_data_from_json(input_file="/notebooks/json/extracted_pages_data.json")
print(f"{len(pages_data)} Seiten wurden geladen.")

# %%
inspect_pages_data(pages_data)


# %%
# Beispiel: Wandele pages_data um in LangChain documents
documents = build_documents_from_pages(pages_data)


# %%
# Jetzt Text in Chunks splitten
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
alle_chunks = text_splitter.split_documents(documents)
    
print(f"Aus {len(documents)} Seiten wurden {len(alle_chunks)} Chunks erzeugt.")


# %%
for i, chunk in enumerate(alle_chunks):
    print(f"\n=== Chunk {i+1} ===")
    print(chunk.page_content[:500] + "...")
    print("Seite:", chunk.metadata.get("page_number"))

# %%
# Embeddings mit HuggingFace Sentence Transformers (MiniLM)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# %%
# VectorStore erzeugen
vectorstore = FAISS.from_documents(alle_chunks, embeddings)

# %%
# VectorStore speichern
speicherpfad = "/notebooks/vectorstore/"
vectorstore.save_local(speicherpfad)
print(f"Vektordatenbank gespeichert unter: {speicherpfad}")

# %%
# Vectorstore laden
# speicherpfad = "/notebooks/vectorstore/"
# vectorstore = load_vectorstore(speicherpfad)

# %%
# Lokales LLM initialisieren
llm = LocalLlamaLLM(device="gpu")

# %%
# RetrievalQA Chain bauen (gemeinsam für alle PDFs)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# %%
# Frage
frage = "Extrahiere aus dem Dokument alle Informationen zu THG-Emissionen!"
antwort = qa_chain.invoke({"query": frage})
print("Antwort:", antwort)

# %%
# Frage
frage = "Wie hoch sind die Scope-1-THG-Bruottomissionen im Jahr 2024!"
antwort = qa_chain.invoke({"query": frage})
print("Antwort:", antwort)

# %%
# Dein eigener Prompt mit der Einschränkung
custom_prompt_template = """Beantworte die folgende Frage basierend auf dem folgenden Kontext.
Suche alle Zahlen in dem Text, die einen Nachhaltigkeitsbezug haben

Kontext: {context}

Frage: {question}
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=custom_prompt_template
)

# 2. RetrievalQA mit eigenem Prompt erstellen
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# %%
# Frage
frage = "Gebe für jede im Dokument gefundene Zahl zu Emissionen mit einem von Dir erkannten Bezug folgendes aus: {Bezug, Zahl} Keine weitere Ausgabe!"
antwort = qa_chain.invoke({"query": frage})
print("Antwort:", antwort)

# %%
print_clean_result (antwort)
