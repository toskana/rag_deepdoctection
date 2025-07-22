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

# %%
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import os


# %%
# --- Eigener LLM Wrapper für llama.cpp HTTP Service ---
class LocalLlamaLLM(LLM):
    endpoint: str = "http://llm:5000/completion"

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
# PDF laden und Text extrahieren
loader = PyPDFLoader("sample.pdf")  # Pfad zu deiner PDF
documents = loader.load()[0]

# Text in kleinere Chunks splitten (damit die Vektor-Einbettung besser funktioniert)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents([documents])

print(f"Dokument geladen und in {len(texts)} Chunks aufgeteilt.")

# %%
# 1. Pfad zur PDF-Sammlung
pdf_ordner = "./pdf/"

# 2. Alle PDF-Dateien im Verzeichnis finden
pdf_dateien = [f for f in os.listdir(pdf_ordner) if f.lower().endswith(".pdf")]

print(f"{len(pdf_dateien)} PDF-Dateien gefunden.")

# 3. Chunking vorbereiten
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# %%
# 4. Schleife über alle PDFs
for dateiname in pdf_dateien:

    pfad_zur_datei = os.path.join(pdf_ordner, dateiname)
    print(f"Verarbeite Datei: {dateiname}", "\n____________________________________________________________________________")

    # PDF laden
    loader = PyPDFLoader(pfad_zur_datei)
    dokument = loader.load()[0]

    # In Chunks aufteilen
    texts = text_splitter.split_documents([dokument])      

    # Embeddings mit HuggingFace Sentence Transformers (MiniLM)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # FAISS Vectorstore aus den Dokumenten erstellen
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Lokales LLM initialisieren
    llm = LocalLlamaLLM()

    # RetrievalQA Chain bauen
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    # Autor
    frage = "Nenne den Namen des Autors des Dokuments. Gib keine weiteren Informationen zurück."
    antwort = qa_chain.run(frage)
    print("Name:", antwort)

    # Matrikelnummer
    frage = "Nenne die Matrikelnummer auf dem Deckblatt, das ist eine mehrstellige Zahl"
    antwort = qa_chain.run(frage)
    print("Matrikelnummer:", antwort)

    # Titel
    frage = "Nenne den vollständigen Titel des Dokuments. Das Wort Masterarbeit oder Bachelorarbeit gehört nicht zum Titel."
    antwort = qa_chain.run(frage)
    print("Titel:", antwort)

    # Studiengang
    frage = (
        """Studiengänge beginnen am Anfang häufig mit MA, MBA, BA, Master, Bachelor.
        Gib ausschließlich den offiziellen Namen des Studiengangs zurück.
        Gib **keine weiteren Informationen** zurück. 
        **Keine Korrekturen, keine Erklärungen, keine Beurteilungen.**"""
            )
    antwort = qa_chain.run(frage)
    print("Studiengang:", antwort)

    # Bachelor, Master oder MBA
    frage = ("""Handelt es sich bei dem Studiengang um einen MBA, dann gib {MBA} zurück.
             In allen anderen Fällen gib {BA/MA} zurück.
             Gebe keine anderen Antworten außer {MBA} und {BA/MA}"""
            )
    antwort = qa_chain.run(frage)
    print("MBA oder BA/MA:", antwort, "\n\n")


# %%
# 1. Dein eigener Prompt mit der Einschränkung
custom_prompt_template = """Beantworte die folgende Frage basierend auf dem folgenden Kontext.
Keine weiteren Ausgaben.
Gib nur eine JSON-Antwort mit folgendem Format zurück:
{{ "antwort": "..." }}

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
# Frage Autor an das PDF
frage = "Nenne den Namen des Autors des Dokuments."
antwort = qa_chain.run(frage)
print("Antwort:", antwort)

# %%
