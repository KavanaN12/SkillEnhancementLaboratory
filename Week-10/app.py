import os
import re
from pathlib import Path
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

from google import genai
from google.genai import types

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vectorstore" / "ipc_faiss"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IPC RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #777;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-heading {
        font-size: 30px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .small-note {
        color: #777;
        font-size: 14px;
        font-style: italic;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #dddddd;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## About")

    st.write(
        "This chatbot retrieves relevant passages from the supplied "
        "Indian Penal Code document before asking Gemini to answer "
        "from that context."
    )

    st.markdown("**Embedding model:** `all-MiniLM-L6-v2`")

    st.markdown("**Vector store:** FAISS")

    st.markdown(f"**LLM:** `{GEMINI_MODEL}`")

    st.markdown("**Framework:** LangChain")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# LOAD VECTOR STORE
# ============================================================

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db = FAISS.load_local(
        str(VECTOR_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return db


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini():

    if not GEMINI_API_KEY:
        return None

    return genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# BASIC VALIDATION
# ============================================================

if not VECTOR_DIR.exists():

    st.error(
        "FAISS vector store was not found. "
        "Please run `python ingest.py` first."
    )

    st.stop()


if not GEMINI_API_KEY:

    st.error(
        "GEMINI_API_KEY is missing. "
        "Add it to your .env file and restart Streamlit."
    )

    st.stop()


try:

    db = load_vectorstore()
    gemini_client = load_gemini()

except Exception as e:

    st.error(f"Unable to load the chatbot resources: {e}")

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_section_number(text: str):
    """
    Extract an IPC section number from the user's question.

    Examples:
        Section 302
        section 304
        IPC 420
    """

    patterns = [
        r"\bsection\s+(\d{1,3}[A-Z]?)\b",
        r"\bsec\.?\s+(\d{1,3}[A-Z]?)\b",
        r"\bIPC\s+(\d{1,3}[A-Z]?)\b",
    ]

    for pattern in patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1)

    return None


def retrieve_documents(question: str):
    """
    Retrieve relevant IPC passages.

    MMR is used to reduce near-duplicate chunks.
    """

    section_number = extract_section_number(question)

    try:

        # Retrieve a slightly larger candidate set first.
        documents = db.max_marginal_relevance_search(
            question,
            k=8,
            fetch_k=24,
        )

    except Exception:

        documents = db.similarity_search(
            question,
            k=8,
        )

    # --------------------------------------------------------
    # If the user explicitly mentions a section, prioritize
    # documents carrying that section metadata.
    # --------------------------------------------------------

    if section_number:

        matching = []
        others = []

        for doc in documents:

            doc_section = str(
                doc.metadata.get("section", "")
            ).strip()

            if doc_section == section_number:
                matching.append(doc)
            else:
                others.append(doc)

        documents = matching + others

    # --------------------------------------------------------
    # Remove duplicate chunks.
    # --------------------------------------------------------

    unique_documents = []
    seen = set()

    for doc in documents:

        text = doc.page_content.strip()

        # Normalize enough to identify duplicates.
        key = re.sub(r"\s+", " ", text).lower()

        if key in seen:
            continue

        seen.add(key)
        unique_documents.append(doc)

    return unique_documents[:5]


def get_page_number(doc):

    page = doc.metadata.get("page")

    if page is None:
        return "Unknown"

    try:
        return int(page) + 1

    except Exception:
        return page


def get_section(doc):

    section = doc.metadata.get("section")

    if section:
        return str(section)

    return "General"


def build_context(documents):

    context_parts = []

    for i, doc in enumerate(documents, start=1):

        section = get_section(doc)
        page = get_page_number(doc)

        text = doc.page_content.strip()

        # Keep individual chunks reasonably small.
        text = text[:3500]

        context_parts.append(
            f"""
SOURCE {i}
IPC SECTION: {section}
PDF PAGE: {page}

{text}
"""
        )

    # Prevent unnecessarily large Gemini requests.
    context = "\n\n".join(context_parts)

    return context[:12000]


def build_history():

    """
    Build a short conversational history.

    Only the latest few exchanges are included so that
    follow-up questions remain understandable without
    unnecessarily increasing Gemini token usage.
    """

    if not st.session_state.messages:
        return ""

    recent = st.session_state.messages[-6:]

    history_parts = []

    for message in recent:

        role = message["role"].upper()

        content = message["content"]

        # Avoid sending extremely long previous responses.
        content = content[:1800]

        history_parts.append(
            f"{role}: {content}"
        )

    return "\n".join(history_parts)


def generate_answer(question: str, context: str):

    history = build_history()

    system_instruction = """
You are an IPC document-grounded question answering assistant.

Your task is to answer questions using ONLY the supplied context
retrieved from the provided Indian Penal Code document.

Rules:

1. Do not invent IPC sections, provisions, punishments,
   definitions, classifications or case-law details.

2. Prefer the statutory provision over commentary when answering
   what a section says.

3. Clearly distinguish:
   - Main statutory provision
   - Classification of offence
   - Comments / case-law

4. If the user asks a follow-up such as "what is its punishment?",
   use the conversation history to determine which section
   "its" refers to.

5. If the requested information is not present in the supplied
   context, clearly say that it could not be located in the
   retrieved document context.

6. Keep answers structured and easy to understand.

7. Mention the relevant IPC section and PDF page whenever possible.

8. Do not provide legal advice. This is an educational
   document-retrieval system.
"""

    prompt = f"""
CONVERSATION HISTORY:
{history}

CURRENT USER QUESTION:
{question}

RETRIEVED IPC DOCUMENT CONTEXT:
{context}

Answer the current question using the retrieved context.
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                max_output_tokens=1000,
            ),
        )

        return response.text.strip()

    except Exception as e:

        error_text = str(e).lower()

        # ----------------------------------------------------
        # Gemini rate-limit handling
        # ----------------------------------------------------

        if (
            "429" in error_text
            or "resource_exhausted" in error_text
            or "rate limit" in error_text
            or "quota" in error_text
        ):

            return (
                "⚠️ **Gemini API rate limit reached temporarily.**\n\n"
                "The document retrieval part of the system is still "
                "working correctly, but Gemini cannot generate another "
                "response at the moment.\n\n"
                "**For demonstration:** the relevant IPC passages "
                "retrieved for this question are shown below in "
                "`Retrieved IPC sources`.\n\n"
                "Please wait briefly before sending another Gemini "
                "request."
            )

        # ----------------------------------------------------
        # Authentication
        # ----------------------------------------------------

        if (
            "api key" in error_text
            or "permission_denied" in error_text
            or "unauthenticated" in error_text
        ):

            return (
                "⚠️ **Gemini API authentication error.**\n\n"
                "Please check the GEMINI_API_KEY in your `.env` file."
            )

        # ----------------------------------------------------
        # Other API errors
        # ----------------------------------------------------

        return (
            "⚠️ **Gemini could not generate a response.**\n\n"
            f"Technical detail: `{str(e)[:300]}`"
        )


def display_sources(documents):

    with st.expander(
        "Retrieved IPC sources",
        expanded=False,
    ):

        for i, doc in enumerate(documents, start=1):

            section = get_section(doc)
            page = get_page_number(doc)

            st.markdown(
                f"### Source {i} — Section {section} — PDF Page {page}"
            )

            st.caption(
                doc.metadata.get(
                    "section_title",
                    "IPC document passage"
                )
            )

            st.write(doc.page_content)

            if i < len(documents):
                st.divider()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Indian Penal Code — RAG Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Document-grounded conversational QA using LangChain, FAISS, '
    'HuggingFace embeddings and Google Gemini'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):

            st.markdown(
                "**Disclaimer:** "
                "*This is an educational document-retrieval system, "
                "not legal advice.*"
            )

            st.markdown(message["content"])

            if message.get("documents"):

                display_sources(
                    message["documents"]
                )


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask about an IPC section, offence, punishment, definition, or provision..."
)


if question:

    # --------------------------------------------------------
    # Display user question immediately.
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # --------------------------------------------------------
    # Retrieval happens locally.
    # --------------------------------------------------------

    documents = retrieve_documents(question)

    if not documents:

        answer = (
            "I could not find a relevant passage in the supplied "
            "Indian Penal Code document."
        )

    else:

        context = build_context(documents)

        # ----------------------------------------------------
        # Gemini generation
        # ----------------------------------------------------

        with st.spinner("Retrieving IPC context and generating answer..."):

            answer = generate_answer(
                question,
                context,
            )

    # --------------------------------------------------------
    # Store assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "documents": documents,
        }
    )

    # --------------------------------------------------------
    # Display response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(
            "**Disclaimer:** "
            "*This is an educational document-retrieval system, "
            "not legal advice.*"
        )

        st.markdown(answer)

        if documents:
            display_sources(documents)