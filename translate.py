import os
import asyncio
import ollama
import nbformat


TRANSLATE_MODEL = "translategemma:12b"
example_nb_path = "01_intro.ipynb"
example_text = text = (
    "> note: Full and Stripped Notebooks: There are two folders containing "
    "different versions of the notebooks. The _full_ folder contains the exact "
    "notebooks used to create the book you're reading now, with all the prose "
    "and outputs. The _stripped_ version has the same headings and code cells, "
    "but all outputs and prose have been removed. After reading a section of "
    "the book, we recommend working through the stripped notebooks, with the "
    "book closed, and seeing if you can figure out what each cell will show "
    "before you execute it. Also try to recall what the code is demonstrating."
)


async def translate_cell(
        index,
        cell,
        src_lang_name,
        src_lang_code,
        target_lang_name,
        target_lang_code
        ):
    if cell.cell_type == "code":
        return cell
    elif cell.cell_type in ["markdown", "raw"]:
        prompt_template = (
            "You are a professional {src_lang_name} ({src_lang_code}) to "
            "{target_lang_name} ({target_lang_code}) translator. "
            "Your goal is to accurately convey the meaning and nuances of the "
            "original English text while adhering to {target_lang_name} "
            "grammar, vocabulary, and cultural sensitivities.\n"
            "Produce only the {target_lang_name} translation, "
            "without any additional explanations or commentary. "
            "Preserve the original Markdown formatting of the text. "
            "Please translate the following {src_lang_name} text into "
            "{target_lang_name}:\n\n\n{text}\n"
        )
        prompt = prompt_template.format(
            src_lang_name=src_lang_name,
            src_lang_code=src_lang_code,
            target_lang_name=target_lang_name,
            target_lang_code=target_lang_code,
            text=cell.source,
        )
        ollama_client = ollama.AsyncClient()
        resp = await ollama_client.generate(model=TRANSLATE_MODEL, prompt=prompt)
        print(f"{index} - {cell.cell_type}")
        new_cell = (nbformat.v4.new_markdown_cell(resp["response"])
                    if cell.cell_type == "markdown"
                    else nbformat.v4.new_raw_cell(resp["response"]))
        return new_cell


def translate_notebook(
        src_notebook_path,
        dest_notebook_path,
        src_lang_name="English",
        src_lang_code="en",
        target_lang_name="Russian",
        target_lang_code="ru"
        ):

    dest_nb = nbformat.v4.new_notebook()

    with open(src_notebook_path, "r", encoding="utf-8") as src_file:
        print(f"Reading {src_notebook_path}")
        src_nb = nbformat.read(src_file, as_version=4)
        cells_count = len(src_nb.cells)
        print(f"Cell count: {cells_count}")

        tasks = []
        for index, src_cell in enumerate(src_nb.cells, start=1):
            tasks.append(
                translate_cell(
                    index,
                    src_cell,
                    src_lang_name,
                    src_lang_code,
                    target_lang_name,
                    target_lang_code,
                )
            )

        async def translate_main():
            return await asyncio.gather(*tasks)

        results = asyncio.run(translate_main())
        dest_nb.cells.extend(results)

    with open(dest_notebook_path, "w", encoding="utf-8") as dest_file:
        nbformat.write(dest_nb, dest_file)
        print(f"Saved to file {dest_notebook_path}")


def translate_notebooks(ipynb_files, destination_directory="translations/ru"):
    for notebook_path in ipynb_files:
        destination_path = os.path.join(destination_directory, notebook_path)
        translate_notebook(notebook_path, destination_path)
