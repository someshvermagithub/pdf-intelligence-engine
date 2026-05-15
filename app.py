import os
import shutil
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------
# Load Environment Variables
# ---------------------------------

load_dotenv()


# ---------------------------------
# Streamlit Config
# ---------------------------------

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Research Assistant")


# ---------------------------------
# Session State
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False


# ---------------------------------
# Sidebar Controls
# ---------------------------------

st.sidebar.header("⚙️ Settings")

chunk_size = st.sidebar.slider(
    "Chunk Size",
    500,
    2000,
    1000
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    0,
    500,
    200
)

k_value = st.sidebar.slider(
    "Retriever k",
    1,
    10,
    4
)

temperature = st.sidebar.slider(
    "LLM Temperature",
    0.0,
    1.0,
    0.2
)


# ---------------------------------
# Reset Database
# ---------------------------------

if st.sidebar.button("🗑 Reset Database"):

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    st.session_state.vectorstore_ready = False

    st.sidebar.success("Database reset complete")


# ---------------------------------
# Cached Models
# ---------------------------------

@st.cache_resource
def load_embedding_model():

    return MistralAIEmbeddings(
        model="mistral-embed"
    )


@st.cache_resource
def load_llm(temp):

    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=temp
    )


embedding_model = load_embedding_model()

llm = load_llm(temperature)


# ---------------------------------
# File Upload
# ---------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ---------------------------------
# Create Vector Database
# ---------------------------------

if uploaded_files:

    if st.button("Create Vector Database"):

        with st.spinner("Processing PDFs..."):

            documents = []

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(
                        uploaded_file.read()
                    )

                    file_path = tmp_file.name

                loader = PyPDFLoader(file_path)

                docs = loader.load()

                documents.extend(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = splitter.split_documents(
                documents
            )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db"
            )

            vectorstore.persist()

            st.session_state.vectorstore_ready = True

        st.success("Vector Database Created!")


# ---------------------------------
# Load Vector Store
# ---------------------------------

if os.path.exists("chroma_db"):

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k_value,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # ---------------------------------
    # Prompt Template
    # ---------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert AI research assistant.

Rules:
- Use ONLY provided context
- Be accurate and concise
- If answer is unavailable, say:
"I could not find the answer in the document."
"""
            ),

            (
                "human",
                """
Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()

    st.subheader("💬 Chat With Your PDFs")

    query = st.chat_input(
        "Ask something about the document..."
    )

    if query:

        st.session_state.messages.append(
            ("user", query)
        )

        with st.spinner("Thinking..."):

            docs = retriever.invoke(query)

            context = "\n\n".join([
                doc.page_content
                for doc in docs
            ])

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = llm.invoke(final_prompt)

            answer = response.content

        st.session_state.messages.append(
            ("assistant", answer)
        )

    # ---------------------------------
    # Display Chat History
    # ---------------------------------

    for role, message in st.session_state.messages:

        with st.chat_message(role):

            st.write(message)

    # ---------------------------------
    # Retrieved Sources
    # ---------------------------------

    if query:

        with st.expander(
            "📄 Retrieved Source Chunks"
        ):

            for i, doc in enumerate(docs):

                st.markdown(
                    f"### Chunk {i+1}"
                )

                st.write(
                    doc.page_content
                )

                st.divider()