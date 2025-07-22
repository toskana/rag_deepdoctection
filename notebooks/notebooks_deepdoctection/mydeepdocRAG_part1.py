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
import torch
print(torch.cuda.is_available())

# %%
import deepdoctection as dd
import pandas as pd
from typing import List, Dict, Any
import json


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
