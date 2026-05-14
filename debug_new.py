"""Debug version of the new code"""
import streamlit as st

st.set_page_config(page_title="Test", page_icon="📄")
st.title("🚀 AI Resume + Job Matching System")

st.write("Step 1: Basic UI ✅")

# Test each part
try:
    st.write("Step 2: Testing load_dotenv...")
    from dotenv import load_dotenv
    load_dotenv()
    st.write("load_dotenv ✅")
except Exception as e:
    st.error(f"load_dotenv failed: {e}")

try:
    st.write("Step 3: Testing json, logging, numpy...")
    import json, logging, numpy as np
    st.write("json, logging, numpy ✅")
except Exception as e:
    st.error(f"Basic imports failed: {e}")

try:
    st.write("Step 4: Testing sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    st.write("sentence_transformers import ✅")
    
    st.write("Step 5: Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("SentenceTransformer model ✅")
except Exception as e:
    st.error(f"sentence_transformers failed: {e}")

try:
    st.write("Step 6: Testing langchain...")
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    st.write("langchain imports ✅")
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    st.write("ChatOpenAI ✅")
except Exception as e:
    st.error(f"langchain failed: {e}")

try:
    st.write("Step 7: Testing pdfplumber...")
    import pdfplumber
    st.write("pdfplumber ✅")
except Exception as e:
    st.error(f"pdfplumber failed: {e}")

try:
    st.write("Step 8: Testing pypdf...")
    from pypdf import PdfReader
    st.write("pypdf ✅")
except Exception as e:
    st.error(f"pypdf failed: {e}")

st.success("All steps complete!")