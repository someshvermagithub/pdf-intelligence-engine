import os
import shutil
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------
# Load Environment Variables
# ---------------------------------
load_dotenv()

# ---------------------------------
# Streamlit Config & Custom CSS
# ---------------------------------
st.set_page_config(
    page_title="DocuMind AI | Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean SaaS look (hides default Streamlit branding)
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
 
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Style the chat input area to look more modern */
        .stChatInput {
            padding-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# Session State Initialization
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = os.path.exists("chroma_db")
if "last_context" not in st.session_state:
    st.session_state.last_context = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False

# ---------------------------------
# Cached Models
# ---------------------------------
@st.cache_resource
def load_embedding_model():
    return MistralAIEmbeddings(model="mistral-embed")

@st.cache_resource
def load_llm(temp):
    return ChatMistralAI(model="mistral-small-latest", temperature=temp)

embedding_model = load_embedding_model()

# ---------------------------------
# Sidebar Workspace (Settings & Uploads)
# ---------------------------------
with st.sidebar:
    st.title("🧠 DocuMind AI")
    st.caption("Your intelligent research workspace")
    st.divider()

    st.subheader("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to create your knowledge base."
    )

    if uploaded_files:
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            with st.status("Building Knowledge Base...", expanded=True) as status:
                st.write("Reading PDFs...")
                documents = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        file_path = tmp_file.name

                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)

                st.write("Chunking text...")
                splitter = RecursiveCharacterTextSplitter(
                    # Using default values here; can be overridden by advanced settings
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(documents)

                st.write("Generating embeddings...")
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory="chroma_db"
                )
                vectorstore.persist()
                
                st.session_state.vectorstore_ready = True
                status.update(label="Knowledge Base Ready!", state="complete", expanded=False)
            
            st.toast("Documents processed successfully!", icon="✅")
            st.rerun()

    st.divider()

    # Hide technical sliders inside an expander for a cleaner UI
    with st.expander("⚙️ Advanced Settings"):
        k_value = st.slider("Retrieval Count (k)", 1, 10, 4, help="How many text chunks to pull for context.")
        temperature = st.slider("Creativity (Temp)", 0.0, 1.0, 0.2, help="Lower is more factual, higher is more creative.")
        
        if st.button("🗑️ Clear Database", use_container_width=True):
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")
            st.session_state.vectorstore_ready = False
            st.session_state.messages = [] # Clear chat too
            st.toast("Database and chat history cleared.", icon="🗑️")
            st.rerun()

llm = load_llm(temperature)

# ---------------------------------
# Main Chat Interface
# ---------------------------------

# Empty State / Onboarding Screen
if not st.session_state.vectorstore_ready:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👋 **Welcome to your AI Research Assistant!**", icon="✨")
        st.markdown("""
        To get started:
        1. Open the sidebar on the left.
        2. Upload your PDF documents.
        3. Click **Process Documents**.
        4. Start asking questions!
        """)
else:
    # Active Workspace
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k_value, "fetch_k": 10, "lambda_mult": 0.5}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI research assistant. 
        Rules:
        - Use ONLY provided context
        - Be accurate, structured, and professional. Use markdown formatting (bolding, lists) to make answers scannable.
        - If answer is unavailable, say: "I could not find the answer in the provided documents."
        """),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    # Display Chat History
    for role, message in st.session_state.messages:
        with st.chat_message(role):
            st.write(message)

    # Feedback Mechanism (Shown only for the most recent answer)
    if st.session_state.show_feedback and len(st.session_state.messages) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        feedback_container = st.container()
        with feedback_container:
            st.caption("Was this answer helpful?")
            col1, col2, col3 = st.columns([1, 1, 8])
            
            with col1:
                if st.button("👍 Yes", key="upvote", use_container_width=True):
                    st.toast("Thanks for the feedback!", icon="🎉")
                    st.session_state.show_feedback = False
                    st.rerun()
                    
            with col2:
                if st.button("👎 No", key="downvote", use_container_width=True):
                    st.session_state.show_feedback = False 
                    st.toast("Working on a better explanation...", icon="🔄")
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Refining answer..."):
                            refine_prompt = ChatPromptTemplate.from_messages([
                                ("system", "You are an expert AI. The user wasn't satisfied with the last answer. Using ONLY the provided context, provide a much clearer, more detailed, and heavily structured (bullet points/bold text) explanation. Break down complex topics step-by-step."),
                                ("human", "Context:\n{context}\n\nQuestion:\n{question}")
                            ])
                            
                            final_refine_prompt = refine_prompt.invoke({
                                "context": st.session_state.last_context,
                                "question": st.session_state.last_query
                            })
                            
                            new_response = llm.invoke(final_refine_prompt)
                            improved_answer = "*(Regenerated for better clarity)*\n\n" + new_response.content
                            
                            st.write(improved_answer)
                            st.session_state.messages.append(("assistant", improved_answer))
                            st.rerun()

    # Chat Input
    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state.show_feedback = False 
        
        # Add user message to UI immediately
        st.session_state.messages.append(("user", query))
        with st.chat_message("user"):
            st.write(query)

        # Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                st.session_state.last_context = context
                st.session_state.last_query = query

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": query
                })

                response = llm.invoke(final_prompt)
                answer = response.content
                st.write(answer)
                
                # Show sources inside an expander attached to the message
                with st.expander("View Source Context"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}**")
                        st.caption(doc.page_content)
                        st.divider()

        st.session_state.messages.append(("assistant", answer))
        st.session_state.show_feedback = True
        st.rerun()