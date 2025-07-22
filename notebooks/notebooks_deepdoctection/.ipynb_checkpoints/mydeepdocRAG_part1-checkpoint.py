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
# ---

# %%
import deepdoctection as dd
import pandas as pd
from typing import List, Dict, Any

# %%
import json


# %%
def table_to_dataframe(table) -> pd.DataFrame:
    """Konvertiert eine DeepDoctection-Tabelle in ein Pandas-DataFrame."""
    n_rows = table.number_of_rows
    n_cols = table.number_of_columns
    cells = table.cells

    cell_map = {}
    for cell in cells:
        r = getattr(cell, "row_number", None)
        c = getattr(cell, "column_number", None)
        text = getattr(cell, "text", "")
        if r is not None and c is not None:
            cell_map[(r, c)] = text

    rows = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            row.append(cell_map.get((r, c), ""))
        rows.append(row)

    return pd.DataFrame(rows)


# %%
def extract_page_data(page) -> Dict[str, Any]:
    """Extrahiert alle wichtigen Inhalte aus einer einzelnen Seite."""
    page_info = {
        "page_number": page.page_number,
        "file_name": page.file_name,
        "document_id": page.document_id,
        "image_id": page.image_id,
        "width": page.width,
        "height": page.height,
        "text": page.text,
        "chunks": [],
        "tables": []
    }

    # Text-Chunks (aus Layoutanalyse)
    if page.chunks:
        for chunk in page.chunks:
            if isinstance(chunk, tuple) and len(chunk) == 2:
                bbox, text = chunk
                page_info["chunks"].append({
                    "bbox": bbox,
                    "text": text
                })

    # Tabellen als DataFrames
    if page.tables:
        for table in page.tables:
            df_table = table_to_dataframe(table)
            page_info["tables"].append(df_table)

    return page_info


# %%
def analyze_pdf(path: str) -> List[Dict[str, Any]]:
    """Analysiert das PDF und liefert eine Liste von Seiteninformationen."""
    analyzer = dd.get_dd_analyzer()
    df = analyzer.analyze(path=path)

    pages = []
    for page in df:
        page_data = extract_page_data(page)
        pages.append(page_data)

    df.reset_state()
    return pages


# %%
def inspect_pages_data(pages_data, max_pages=2, max_chunks=3, max_text_len=100):
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
        # Nur Ausgabe des Typs der Tabellen, da DataFrames komplex sind
        for k, table in enumerate(tables[:max_chunks]):
            print(f"  Tabelle {k+1}: Typ={type(table)}")


# %%
def export_pages_data_as_json(pages_data, output_file):
    output = []
    for page in pages_data:
        page_entry = {
            "page_number": page.get("page_number"),
            "file_name": page.get("file_name"),
            "document_id": page.get("document_id"),
            "image_id": page.get("image_id"),
            "width": page.get("width"),
            "height": page.get("height"),
            "text": page.get("text"),
            "tables": []
        }

        for df in page.get("tables", []):
            csv_str = df.to_csv(index=False)
            page_entry["tables"].append(csv_str)

        output.append(page_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# %%
PDF_PATH = "/repo/notebooks/data/2024-nachhaltigkeitsbericht_tab.pdf"
pages_data = list(analyze_pdf(PDF_PATH))
print(f"{len(pages_data)} Seiten analysiert.")


# %%
inspect_pages_data (pages_data)


# %%
print(f"Anzahl Seiten in pages_data: {len(pages_data)}")
for i, page in enumerate(pages_data):
    print(f"Seite {i} keys: {list(page.keys())}")
    print(f"Text länge: {len(page['text'])}")
    print(f"Tabellenanzahl: {len(page['tables'])}")
    if len(page['tables']) > 0:
        print(f"Erste Tabelle (head):\n{page['tables'][0].head()}")

# %%
OUTPUT_JSON = "/repo/notebooks/json/extracted_pages_data.json"  # Beispiel Pfad im Shared Volume
export_pages_data_as_json (pages_data, OUTPUT_JSON)

# %%
# !pip install langchain

# !pip install langchain-community

# #!pip install sentence-transformers

# %%
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import json
import pandas as pd
from io import StringIO


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
def inspect_pages_data(pages_data, max_pages=2, max_chunks=3, max_text_len=100):
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
        # Nur Ausgabe des Typs der Tabellen, da DataFrames komplex sind
        for k, table in enumerate(tables[:max_chunks]):
            print(f"  Tabelle {k+1}: Typ={type(table)}")


# %%
with open("/repo/notebooks/json/extracted_pages_data.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

# %%
for page in pages:
    print(f"Seite {page['page_number']}, Datei: {page['file_name']}")
    print(f"Text snippet: {page['text'][:100]}...\n")

    for i, csv_table in enumerate(page["tables"]):
        df = pd.read_csv(StringIO(csv_table))
        print(f"Tabelle {i+1}:")
        print(df.head())
        print()

# %%
print(f"pages_data type: {type(pages_data)}")
print(f"Anzahl Seiten: {len(pages_data)}")
if len(pages_data) > 0:
    print("Keys der ersten Seite:", pages_data[0].keys())

inspect_pages_data(pages_data)


# %%
# Beispiel: Nutze die Funktion mit dem DeepDoctection-Output (pages_data)
documents = build_documents_from_pages(pages_data)
    
# Jetzt Text in Chunks splitten
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
    
print(f"Aus {len(documents)} Seiten wurden {len(texts)} Chunks erzeugt.")
    
# texts ist eine Liste von Document-Objekten mit jeweils
# - text_chunk im page_content
# - Metadaten (page_number, file_name)
        


    # %%
    for i, chunk in enumerate(texts):
        print(f"\n=== Chunk {i+1} ===")
        print(chunk.page_content[:300] + "...")
        print("Seite:", chunk.metadata.get("page_number"))

########## Bei Nutzung von Transformers 
#    for i, chunk in enumerate(chunks):
#        print(f"\n=== Chunk {i+1} ===")
#        print(chunk["text"][:300] + "...")
#        print("Seiten:", [p["page_number"] for p in chunk["meta"]])


    


# %%
# Embeddings mit HuggingFace Sentence Transformers (MiniLM)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# %%
model = SentenceTransformer("all-MiniLM-L6-v2")

# Texte aus LangChain-Dokumentobjekten extrahieren
texts_only = [doc.page_content for doc in texts]

# Embeddings erzeugen
embeddings = model.encode(texts_only, show_progress_bar=True)

# Dann mit FAISS verwenden (manuell):
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Originale Metadaten behalten:
vectorstore = FAISS.from_embeddings(
    embeddings=embeddings,
    documents=texts,
)

# %%
# FAISS Vectorstore aus den Dokumenten erstellen
vectorstore = FAISS.from_documents(texts, embeddings)
