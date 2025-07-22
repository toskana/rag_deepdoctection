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
import pandas as pd

# %%
import deepdoctection as dd

# %%
analyzer = dd.get_dd_analyzer()

# %%
df = analyzer.analyze(path="/repo/notebooks/data/2024-nachhaltigkeitsbericht_tab.pdf")

# %%
df.reset_state()

# %%
for dp in df:
    print(dp.text)

# %%
type(dp)

# %%
print(page.get_attribute_names())

# %%
print(f"""height: {page.height}
width: {page.width}
file_name: {page.file_name}
document_id: {page.document_id}
image_id: {page.image_id}
""")

# %%
df.reset_state()

# %%
df = analyzer.analyze(path="/repo/notebooks/data/2024-nachhaltigkeitsbericht_tab.pdf")

# %%
for page in df:
    print(f"=== Seite {page.page_number} ===")
    if page.chunks:
        for chunk in page.chunks:
            if isinstance(chunk, tuple) and len(chunk) == 2:
                bbox, text = chunk
                print(text)
            else:
                print(chunk)  # fallback
    else:
        print("Keine Chunks gefunden.")


# %%
word = page.words[0]
print(word.get_attribute_names())


# %%
for word in page.words:
    # characters ist oft eine Liste einzelner Buchstaben/Zeichen
    if hasattr(word, "characters") and word.characters:
        # Füge die chars zu einem String zusammen
        text = "".join(word.characters)
        print(text)
    elif hasattr(word, "text_line"):
        print(word.text_line)
    else:
        print("Kein Text in Word-Objekt gefunden")


# %%
df.reset_state()

# %%
df = analyzer.analyze(path="/repo/notebooks/data/2024-nachhaltigkeitsbericht_tab.pdf")

# %%
for i, page in enumerate(df, start=1):
    if page.tables:
        print(f"Seite {i} enthält {len(page.tables)} Tabelle(n).")
        # Optional: Beispiel ausgeben, wie eine Tabelle aussieht
        for j, table in enumerate(page.tables):
            print(f"  Tabelle {j+1}:")
            print(table)  # Je nach Typ kann das ein DataFrame oder eine andere Struktur sein
    else:
        print(f"Seite {i} enthält keine Tabellen.")

# %%
print(table.get_attribute_names())


# %%
df.reset_state()

# %%
# 1. Attribute der ersten Cell prüfen
for cell in table.cells:
    print("Cell attributes:", cell.get_attribute_names())
    break

# %%
df.reset_state()

# %%
df = analyzer.analyze(path="/repo/notebooks/data/2024-nachhaltigkeitsbericht_tab.pdf")


# %%
def table_to_dataframe(table):
    n_rows = table.number_of_rows
    n_cols = table.number_of_columns
    cells = table.cells

    # Map (row_number, column_number) -> text
    cells_dict = {}
    for cell in cells:
        r = getattr(cell, "row_number", None)
        c = getattr(cell, "column_number", None)
        text = getattr(cell, "text", "")
        if r is not None and c is not None:
            cells_dict[(r, c)] = text

    # Tabelle als 2D-Liste zusammenbauen
    rows = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            row.append(cells_dict.get((r, c), ""))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

# Anwendung im Loop
for i, page in enumerate(df, start=1):
    if page.tables:
        print(f"Seite {i} hat {len(page.tables)} Tabelle(n).")
        for j, table in enumerate(page.tables):
            print(f"Tabelle {j+1} auf Seite {i}:")
            df_table = table_to_dataframe(table)
            print(df_table)

