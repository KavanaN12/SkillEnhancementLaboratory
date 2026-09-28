# IPC RAG Chatbot — Gemini + LangChain

A document-grounded conversational chatbot for the supplied Indian Penal Code PDF (`data/2023060228.pdf`).

## Architecture

PDF → text extraction → cleaning → chunking → HuggingFace embeddings → FAISS → retrieval → Gemini → Streamlit chatbot

## Technologies

- **PDF loading:** LangChain `PyPDFLoader`
- **Chunking:** `RecursiveCharacterTextSplitter`
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** FAISS
- **LLM:** Google Gemini through LangChain `ChatGoogleGenerativeAI`
- **Interface:** Streamlit

## 1. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS/WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Gemini

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

Then edit `.env`:

```env
GOOGLE_API_KEY=your_new_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

You can use `GEMINI_API_KEY` instead of `GOOGLE_API_KEY` if preferred.

**Never commit `.env` or expose the API key in GitHub.**

## 4. Build the vector store

Run this once:

```bash
python ingest.py
```

This loads `data/2023060228.pdf`, extracts and cleans the text, splits it into chunks, creates local embeddings, and saves the FAISS index under:

```text
vectorstore/ipc_faiss/
```

## 5. Start the chatbot

```bash
streamlit run app.py
```

## Example questions

```text
What is Section 302?
```

```text
What punishment is provided for murder?
```

```text
Explain Section 378 in simple terms.
```

```text
What is the difference between theft and robbery according to the document?
```

You can also test conversational follow-ups:

```text
User: What is Section 302?
Bot: ...
User: What is its punishment?
```

The application rewrites the follow-up question into a standalone retrieval query using recent conversation context before searching FAISS.

## Assignment mapping

1. **Load/preprocess IPC document** → `PyPDFLoader` + cleaning in `ingest.py`
2. **Split into chunks** → `RecursiveCharacterTextSplitter`
3. **Create embedding-based vector store** → HuggingFace embeddings + FAISS
4. **Retrieval-based QA with LLM** → FAISS MMR retrieval + Gemini through LangChain
5. **Conversational chatbot interface** → Streamlit + conversation-aware question rewriting

## Why Gemini is used

Gemini is used only as the generative LLM in this RAG pipeline. The IPC PDF is first processed locally and stored in FAISS. For each question, relevant chunks are retrieved and supplied to Gemini as context. This reduces the chance of the model answering from unrelated knowledge.

The chatbot is an educational document-retrieval system and should not be treated as legal advice.
