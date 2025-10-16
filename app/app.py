import os
import io
import time
import base64
import hashlib
import requests
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

from qa_engine import (
    get_vectorstore,
    build_qa_chain,
    make_retriever,
    get_embeddings,
)
from file_loader import get_raw_text
from ingest_sharepoint import (
    get_graph_access_token,
    detect_link_type,
    fetch_folder_items_from_share_link,
    fetch_file_from_share_link,
    resolve_site_and_list_folder,
    fetch_folder_items_with_site_id,
)

# ---------------- Load Environment ----------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="Doc Chatbot", page_icon="🧠", layout="wide")

# ---------------- CSS Styling ----------------
APP_CSS = """
<style>
/* Custom styles truncated for brevity */
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ---------------- Defaults & Session State ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "settings" not in st.session_state:
    st.session_state.settings = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.2,
        "top_k": 5,
        "chunk_size": 1000,
        "chunk_overlap": 120,
        "persist_dir": os.path.join(os.path.dirname(__file__), "chroma_db"),
        "collection_name": "sharepoint_docs",
        "embedding_model": "text-embedding-3-small",  # OpenAI embeddings
    }

# ---------------- Helpers ----------------
def ensure_vectorstore():
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = get_vectorstore(
            persist_directory=st.session_state.settings["persist_dir"],
            collection_name=st.session_state.settings["collection_name"],
            embedding_model=st.session_state.settings["embedding_model"],
            openai_api_key=OPENAI_API_KEY,
        )
    return st.session_state.vectorstore

def ensure_qa_chain():
    vs = ensure_vectorstore()
    retriever = make_retriever(vs, k=st.session_state.settings["top_k"])
    st.session_state.qa_chain = build_qa_chain(
        retriever=retriever,
        model_name=st.session_state.settings["model_name"],
        temperature=st.session_state.settings["temperature"],
        openai_api_key=OPENAI_API_KEY,
    )
    return st.session_state.qa_chain

def chroma_count(vs):
    try:
        return vs._collection.count()  # may be private API but works widely
    except Exception:
        return None

def add_documents_to_chroma(texts, metadatas, ids):
    vs = ensure_vectorstore()
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    vs.persist()

def clear_index():
    settings = st.session_state.settings
    persist_dir = settings["persist_dir"]
    import shutil
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None

def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

# ---------------- UI Layout ----------------
st.title("🧠 Document Chatbot")

tab_chat, tab_ingest, tab_settings, tab_manage = st.tabs(["💬 Chat", "📥 Ingest (SharePoint)", "⚙️ Settings", "🗂 Manage"])

# --------------- CHAT TAB ---------------
with tab_chat:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Chat with your knowledge base")

    vs = ensure_vectorstore()
    count = chroma_count(vs)
    if count is None or count == 0:
        st.warning("No data in the Chroma index yet. Ingest from SharePoint in the Ingest tab.")
    else:
        st.caption(f"Indexed chunks available: {count}")

    # Chat UI
    chat_placeholder = st.container()
    with chat_placeholder:
        for item in st.session_state.chat_history:
            st.markdown(
                f'<div class="chat-container"><div class="chat-bubble chat-user"><b>You</b><br>{item["q"]}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="chat-container"><div class="chat-bubble chat-bot"><b>Bot</b><br>{item["a"]}</div></div>',
                unsafe_allow_html=True
            )
            if item.get("sources"):
                with st.expander("Sources"):
                    for s in item["sources"]:
                        src = s.metadata.get("source", "unknown")
                        path = s.metadata.get("path", "")
                        lm = s.metadata.get("last_modified", "")
                        st.markdown(f"- {src}  {('('+path+')' if path else '')} {('— '+lm if lm else '')}")
                        st.markdown(f'<span class="small-note">{s.page_content[:300]}...</span>', unsafe_allow_html=True)

    user_query = st.chat_input("Ask something about the indexed documents...")
    if user_query:
        if ensure_vectorstore() and chroma_count(ensure_vectorstore()) not in (None, 0):
            qa = ensure_qa_chain()
            with st.spinner("Thinking..."):
                result = qa.invoke({"query": user_query})
            st.session_state.chat_history.append({
                "q": user_query,
                "a": result["result"],
                "sources": result.get("source_documents", []),
            })
            st.experimental_rerun()
        else:
            st.error("No data available in the index. Please ingest documents first.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------- INGEST TAB ---------------
with tab_ingest:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Ingest from SharePoint into persistent Chroma DB")
    st.caption("Provide a SharePoint file/folder sharing link or site path. Documents will be parsed, chunked, embedded, and persisted in Chroma.")

    sharepoint_url = st.text_input("SharePoint URL (file sharing link, folder sharing link, or site path)")
    col1, col2 = st.columns([1,1])
    with col1:
        clear_before = st.checkbox("Clear index before ingest", value=False)
    with col2:
        only_supported = st.checkbox("Skip unsupported files", value=True)

    if st.button("Ingest Now"):
        if not sharepoint_url.strip():
            st.error("Please enter a SharePoint URL.")
        else:
            if clear_before:
                with st.spinner("Clearing index..."):
                    clear_index()
            ensure_vectorstore()

            with st.spinner("Authenticating with Microsoft Graph..."):
                try:
                    access_token = get_graph_access_token(
                        tenant_id=TENANT_ID,
                        client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET
                    )
                except Exception as e:
                    st.error(f"Auth error: {e}")
                    st.stop()

            link_type = detect_link_type(sharepoint_url)
            all_items = []
            try:
                with st.spinner("Fetching file list..."):
                    if link_type == "folder_share":
                        all_items = fetch_folder_items_from_share_link(sharepoint_url, access_token)
                    elif link_type == "file_share":
                        item = fetch_file_from_share_link(sharepoint_url, access_token, meta_only=True)
                        all_items = [item]
                    else:
                        parsed = urlparse(sharepoint_url)
                        site_hostname = parsed.netloc
                        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
                        if len(path_parts) < 2:
                            raise ValueError("Invalid site path URL. Example: https://<tenant>.sharepoint.com/sites/<site>/**<drive path>**")

                        site_name = path_parts[1]
                        relative_path = "/".join(path_parts[2:])
                        site_id = resolve_site_and_list_folder(site_hostname, site_name, access_token)
                        all_items = fetch_folder_items_with_site_id(site_id, relative_path, access_token)

                if not all_items:
                    st.warning("No files found to ingest.")
                else:
                    chunk_size = st.session_state.settings["chunk_size"]
                    chunk_overlap = st.session_state.settings["chunk_overlap"]
                    texts, metadatas, ids = [], [], []
                    with st.spinner("Downloading, extracting, chunking, and embedding..."):
                        for item in all_items:
                            try:
                                if link_type == "file_share" and not item.get("downloadUrl"):
                                    file_meta = fetch_file_from_share_link(sharepoint_url, access_token, meta_only=False)
                                else:
                                    file_meta = item

                                # Skip folders
                                if file_meta.get("folder"):
                                    continue

                                # Optional filtering by extension
                                name = file_meta.get("name", "unknown")
                                if only_supported and not name.lower().endswith((".pdf", ".docx", ".xlsx", ".xls", ".zip")):
                                    continue

                                download_url = file_meta.get("@microsoft.graph.downloadUrl")
                                if not download_url:
                                    continue
                                res = requests.get(download_url)
                                res.raise_for_status()
                                content_bytes = io.BytesIO(res.content).getvalue()

                                raw_text = get_raw_text(content_bytes, name)
                                if not raw_text.strip():
                                    continue

                                # Chunk
                                from langchain.text_splitter import RecursiveCharacterTextSplitter
                                splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap
                                )
                                chunks = splitter.split_text(raw_text)
                                if not chunks:
                                    continue

                                file_id = file_meta.get("id", name)
                                last_modified = file_meta.get("lastModifiedDateTime", "")
                                server_rel_path = file_meta.get("parentReference", {}).get("path", "")
                                src_path = f"{server_rel_path}/{name}" if server_rel_path else name

                                # Prepare Chroma data
                                for idx, ch in enumerate(chunks):
                                    cid = f"{file_id}:{idx}:{hash_text(ch)}"
                                    metadata = {
                                        "source": name,
                                        "file_id": file_id,
                                        "path": src_path,
                                        "last_modified": last_modified,
                                        "chunk_index": idx,
                                    }
                                    texts.append(ch)
                                    metadatas.append(metadata)
                                    ids.append(cid)
                            except Exception as fe:
                                st.warning(f"Failed processing a file: {fe}")

                        if texts:
                            add_documents_to_chroma(texts, metadatas, ids)
                            st.success(f"Ingested {len(texts)} chunks from {len(all_items)} file(s).")
                        else:
                            st.warning("No chunks were generated (possibly all files unsupported or empty).")

            except requests.HTTPError as he:
                st.error(f"HTTP Error: {he.response.status_code} - {he.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------- SETTINGS TAB ---------------
with tab_settings:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Settings")

    with st.form("settings_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            model_name = st.text_input("LLM model", st.session_state.settings["model_name"])
            top_k = st.number_input("Top K (retrieval)", min_value=1, max_value=20, value=st.session_state.settings["top_k"])
        with c2:
            temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=float(st.session_state.settings["temperature"]), step=0.05)
            chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=st.session_state.settings["chunk_size"], step=50)
        with c3:
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=st.session_state.settings["chunk_overlap"], step=10)
            embedding_model = st.text_input("Embedding model", st.session_state.settings["embedding_model"])

        persist_dir = st.text_input("Chroma persist directory", st.session_state.settings["persist_dir"])
        collection_name = st.text_input("Chroma collection name", st.session_state.settings["collection_name"])

        submitted = st.form_submit_button("Save")
        if submitted:
            st.session_state.settings.update({
                "model_name": model_name,
                "temperature": temperature,
                "top_k": int(top_k),
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "persist_dir": persist_dir,
                "collection_name": collection_name,
                "embedding_model": embedding_model,
            })

            # Reset chain and vectorstore to pick up new settings
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            st.success("Settings saved.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------- MANAGE TAB ---------------
with tab_manage:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Index Management")

    vs = ensure_vectorstore()
    total = chroma_count(vs) or 0
    st.write(f"Current chunks in index: {total}")

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Clear index"):
            clear_index()
            st.success("Index cleared.")
    with c2:
        if st.button("Reload QA chain"):
            ensure_qa_chain()
            st.success("QA chain reloaded.")

    st.caption("Tip: re-run Ingest after clearing the index.")
    st.markdown("</div>", unsafe_allow_html=True)