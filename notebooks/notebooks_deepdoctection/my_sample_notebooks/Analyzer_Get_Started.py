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
# ![title](./pics/dd_logo.png) 
#
#
# # Getting started
#
# **deep**doctection is a package that can be used to extract text from complex structured documents. It also allows you to run vision/text and multi-modal models in an end-to-end pipeline. Inputs can be native PDFs or images. It is very versatile.
#
# Compared to most other parsers, **deep**doctection offers extensive configurability. We will explore these capabilities in more detail in other notebooks.
#
# This notebook will introduce you to the essential basics of document parsing with **deep**doctection.
#
# We assume that, in addition to **deep**doctection, the transformers and python-doctr packages are installed if you are using PyTorch as your deep learning framework. 
#
# If you are using TensorFlow, tensorpack must be installed instead.
#
# Please note that TensorFlow is no longer supported from Python 3.11 onward, and its functionality within **deep**doctection is significantly more limited.
#
# We recommend not using the TensorFlow setup anymore.
#
# You will also need matplotlib that you can install with 
#
# ```
# pip install matplotlib
# ```

# %%
from pathlib import Path
from matplotlib import pyplot as plt
from IPython.core.display import HTML

import deepdoctection as dd

# %% [markdown]
# ## Sample
#
# Take an image (e.g. .png, .jpg, ...). If you take the example below you'll maybe need to change ```image_path```.

# %%
image_path = Path.cwd() / "pics/samples/sample_2/sample_2.png"

# viz_handler is a helper class that helps you e.g. with reading or writing images
image = dd.viz_handler.read_image(image_path)
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)

# %% [markdown]
# ![title](./pics/samples/sample_2/sample_2.png)

# %% [markdown]
# ## Analyzer
#
# Next, we instantiate the **deep**doctection analyzer. The analyzer is an example of a pipeline that can be built depending on the problem you want to tackle. This particular pipeline is built from various building blocks. We will come back to this later. 
#
# We will be using the default configuration.

# %%
analyzer = dd.get_dd_analyzer()

# %% [markdown]
# ## Analyze methods
#
# Once all models have been loaded, we can process a directory with images (.png, .jpf) or multi page PDF-documents. You can either set `path='path/to/dir'` if you have a folder of images or `path='path/to/my/doc.pdf'` if you have a pdf document. 
#
# You will receive an error if your path points to a single image. Processing images requires to pass the path to the base image directory.

# %%
path = Path.cwd() / "/repo/notebooks/data/2024-nachhaltigkeitsbericht_tab.pdf"

df = analyzer.analyze(path=path)

# %% [markdown]
# With
#
# ```
# df = analyzer.analyze(path=path)
# ````
#
# nothing has actually been processed yet.
#
# The method ```analyzer.analyze(path=path)``` does not (yet) return a JSON object, but rather a specialized subclass of the ```DataFlow``` class. Essentially, it behaves like a [generator](https://wiki.python.org/moin/Generators). 
#
# Before starting the iteration, we must call:

# %%
df.reset_state() 

# %% [markdown]
# Now you can traverse through all the values of the `Dataflow` simply by using a `for`-loop or the `next` function. Let's go!  

# %%
doc=iter(df)
page = next(doc)

# %% [markdown]
# ## Page
#
# Let's see what we got back. For each iteration we receive a `Page` object. This object stores all informations that have been collected from a page document when running through the pipeline. 

# %%
type(page)

# %% [markdown]
# Let's also have a look on some top level information. 

# %%
print(f" height: {page.height} \n width: {page.width} \n file_name: {page.file_name} \n document_id: {page.document_id} \n image_id: {page.image_id}\n")

# %% [markdown]
# `document_id` and `image_id` are the same. The reason is because we only process a single image. The naming convention silently assumes that we deal with a one page document. Once we process multi page PDFs `document_id` and `image_id` differ.
#
# With `get_attribute_names()` you get a list of all attributes. 

# %%
page.get_attribute_names()

# %%
page.document_type, page.language

# %% [markdown]
# `page.document_type` and `page.language` both return None. The reason is that the analyzer has no component for predicting a document type or a language. If you want that, you need to build a custom pipeline. Check this [notebook](Using_LayoutLM_for_sequence_classification.ipynb) for further information.
#
# ## Layout segments
#
# We can visualize detected layout segments. If you set `interactive=True` a viewer will pop up. Use `+` and `-` to zoom out/in. Use `q` to close the page.
#
# Alternatively, you can visualize the output with matplotlib.

# %%
image = page.viz()
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)

# %% [markdown]
# ![title](./pics/output_16_1.png)

# %% [markdown]
# Let's have a look at other attributes. We can use the `text` property to get the content of the document. You will notice that the table is not included. You can therefore filter tables from the other content. In fact you can even filter on every layout segment.

# %%
print(page.text)

# %% [markdown]
# You can get the individual layout segments like `text`, `title`, `list` or `figure`. Layout segments also have various attributes. 

# %%
for layout in page.layouts:
    print("--------------")
    print(f"Layout segment: {layout.category_name}, score: {layout.score}, reading_order: {layout.reading_order}, bounding_box: {layout.bounding_box},\n annotation_id: {layout.annotation_id} \n \ntext: {layout.text} \n \n")

# %% [markdown]
# You can also get the layout segments from the `chunks` attribute. The output is a list of tuples with the most essential meta data for each layout segment, namely: `document_id, image_id, page_number, annotation_id, reading_order, category_name` and `text`.

# %%
page.chunks[0]

# %% [markdown]
# Tables cannot be retrieved from `page.layouts`. They have a special `page.tables` which is a python list of table objects. In our situation, only one table has been detected. 

# %%
len(page.tables)

# %% [markdown]
# Let's have a closer look at the table. 

# %%
table = page.tables[0]
table.get_attribute_names()

# %%
print(f" number of rows: {table.number_of_rows} \n number of columns: {table.number_of_columns} \n reading order: {table.reading_order}, \n score: {table.score}")

# %% [markdown]
# There is no reading order. The reason is that we have excluded tables from having a specific reading order position because we want to separate tables from the narrative text. Only layout segments with a `reading_order` not equal to `None`  will be added to the `page.text` string. 
#
# This is pure customizing and we can change the customizing so that tables are part of the narrative text. We will come to this in another tutorial when talking about customization.
#
# You can get an html, csv or text version of the table.  

# %%
HTML(table.html)

# %% [markdown]
# Use `table.csv` to load the table into a Pandas Dataframe.

# %%
table.csv  #pd.DataFrame(table.csv, columns=["Key", "Value"])

# %% [markdown]
# There is also a string representation of the table.

# %%
table.text

# %% [markdown]
# The method `kv_header_rows(row_number)` allows returning column headers and cell contents as key-value pairs for entire rows. Admittedly, the example here is flawed because the table lacks column headers. In fact, the table recognition model determines whether and where a column has a header. In this case, the prediction was incorrect.
#
# However, the principle becomes clear: we receive a dictionary with the schema 
#
# ```{(column_number, column_header(column_number)): cell(row_number, column_number).text}```.

# %%
table.kv_header_rows(2)

# %% [markdown]
# Let's go deeper down the rabbit hole. A `Table` has cells and we can even get the text of one particular cell. Note that the output list is not sorted by row or column. But you can quickly sort the output according to your preferences.

# %%
cell = table.cells[0]
cell.get_attribute_names()

# %%
print(f"column number: {cell.column_number} \n row_number: {cell.row_number}  \n bounding_box: {cell.bounding_box} \n text: {cell.text} \n annotation_id: {cell.annotation_id}")

# %% [markdown]
# Still not down yet, we have a list of `Word`s.  

# %%
for word in cell.words:
    print(f"score: {word.score} \n characters: {word.characters} \n reading_order: {word.reading_order} \n bounding_box: {word.bounding_box}")

# %% [markdown]
# You can see that `Word`s have a `reading_order`, which refers only to the order of the words contained within the `cell`. The `reading_order` applies on two levels: to layout segments within a page and to words within a layout segment.
#
# There are additional reserved attributes, but most of them are not determined by this pipeline.

# %%
word.get_attribute_names()

# %% [markdown]
# ## Saving and reading
#
# You can use the `save` method to save the result of the analyzer in a `.json` file. Setting `image_to_json=True` you will also save image as b64 encoding in the file. Beware, the files can be quite large then. 

# %%
page.save(image_to_json=True, path=Path.cwd() / "pics/samples/sample_2/sample_2.json")

# %% [markdown]
# Having saved the results you can easily parse the file into the `Page` format without loosing any information. The `page` instance below has almost the same structure as the `page` instance returned from the `analyzer` with only some lower level data structure missing that can be reconstructed, though.

# %%
page = dd.Page.from_file(file_path=Path.cwd() / "pics/samples/sample_2/sample_2.json")

# %% [markdown]
# ## Where to go from here
#
# There are several options: 
#
# - You can check this [**notebook**](./Analyzer_More_On_Parsing.ipynb) where we process a different page and explore some more features about the parsed results.
#
# - Maybe you want to switch between different models or want to learn, how you can configure the analyzer. Then the
#   [**configuration notebook**](./Analyzer_Configuration.ipynb) might be interesting.
#
# - If you want to get a deeper understanding how a pipeline is composed, we suggest to look at the [**pipeline notebook**](./Pipelines.ipynb).
