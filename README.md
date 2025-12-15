# RAG Pipeline with PDF Ingestion and Vector Search

This small tool helps you index PDFs and search them using an LLM. It:

- extracts page text from PDFs in `pdfs/` and saves JSON chunks to `jsons/`;
- creates embeddings (via a local BGE-M3 server) and stores them in ChromaDB (`chroma_db/`);
- answers queries by finding the most relevant pages and asking the LLM to produce a helpful reply.

Quick things to know

- Put the PDFs you want to index into the `pdfs/` folder.
- Run `python main.py` and choose an action:
  - `0` — Extract text and save chunks to `jsons/`.
  - `1` — Create embeddings and add them to ChromaDB.
  - `2` — Enter a query to search and get an LLM response.
- The script writes `prompt.txt` and `response.txt` for the last query/response pair.

Adding your own PDFs

You can add your own PDF files by putting them in the `pdfs/` folder and running the extract (`0`) and embed (`1`) steps again. The code works with the included PDFs, and you can test it with your files too. Note that results may vary: the prompt in `main.py` is written for the current course PDFs, so for different kinds of documents you may want to update the prompt or tweak how chunks are created.

Note about included files

PDF files are not included in this repository because they are large. Instead, a few sample extracted files (JSONs) are included so you can test the code quickly. If you want to index your own PDFs, add them to the `pdfs/` folder and run the extract and embedding steps.

Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Environment variables

- `OPENAI_API_KEY` — required to call the OpenAI Responses API.

Troubleshooting

- Make sure `pdfs/` and `jsons/` folders exist and have the expected files.
- If a page's `extract_text()` returns nothing, that page may be an image — try OCR.
- If the embedding server is unreachable, check that the BGE-M3 server is running at `http://localhost:11434`. (Ollama should be installed and BGE-M# model downloaded)
- If OpenAI calls fail, verify `OPENAI_API_KEY` and your network access.
