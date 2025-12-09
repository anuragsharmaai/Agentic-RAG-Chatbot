import os
import json
import streamlit as st
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")

st.title("Multi-Agent Research Assistant")

with st.sidebar:
    st.header("Server Settings")
    api_base = st.text_input("API Base URL", API_BASE)
    st.markdown("Use the backend's base URL, e.g., http://localhost:8000")

st.header("Ingest Documents (RAG)")
up_file = st.file_uploader("Upload PDF", type=["pdf"])
meta_raw = st.text_area("Optional Metadata (JSON)", value="{}")
if st.button("Ingest PDF"):
    if up_file is None:
        st.warning("Please upload a PDF file.")
    else:
        try:
            # Validate metadata JSON but send original string
            _ = json.loads(meta_raw or "{}")
        except Exception:
            st.error("Invalid metadata JSON")
        else:
            with st.spinner("Uploading and ingesting PDF..."):
                try:
                    content = up_file.read()
                    files = {"file": (up_file.name, content, "application/pdf")}
                    data = {"metadata": meta_raw or "{}"}
                    resp = httpx.post(f"{api_base}/api/ingest", files=files, data=data, timeout=120)
                    if resp.status_code == 200:
                        st.success(resp.json())
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))

st.header("Research Query(RAG+Web)")
query = st.text_input("Enter your research question")
col1, col2 = st.columns(2)
with col1:
    max_web = st.number_input("Max Web Results", min_value=1, max_value=10, value=5)
with col2:
    max_rag = st.number_input("Max RAG Chunks", min_value=0, max_value=20, value=5)

if st.button("Run Research"):
    payload = {"query": query, "max_web_results": int(max_web), "max_rag_chunks": int(max_rag)}
    with st.spinner("Researching..."):
        try:
            resp = httpx.post(f"{api_base}/api/research", json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                st.subheader("Executive Summary")
                st.write(data.get("summary", ""))
                st.subheader("Sources")
                for s in data.get("sources", []):
                    st.write(f"- {s}")
                with st.expander("Web Results"):
                    for r in data.get("web_results", []):
                        st.write(r)
                with st.expander("RAG Passages"):
                    for p in data.get("rag_passages", []):
                        st.write(p)
            else:
                st.error(resp.text)
        except Exception as e:
            st.error(str(e))

