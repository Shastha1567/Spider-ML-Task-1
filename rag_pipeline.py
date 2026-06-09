# Shastha Rekha B :)
# Spider ML Task 1 - Applied ML Domain
# RAG Pipeline over NLP Research Papers
#
# What I built here:
# A chatbot that can answer questions about 7 research papers
# by actually searching through them instead of guessing.
# The key idea is RAG (Retrieval Augmented Generation) -
# you find relevant chunks from the papers first, then give
# those chunks to the LLM as context before it answers.
# This way the answers are grounded in actual paper content
# and the LLM tells you which paper it got the info from.
#
# Papers: Attention/Transformers, BERT, GPT-3, RAG, Sentence-BERT, LoRA, Llama 2
#
# Stack: LangChain + ChromaDB + sentence-transformers + Gemini + Gradio
#
# Run in Google Colab - free Gemini API key at aistudio.google.com

# =====================================================================
# CELL 1 - install dependencies
# =====================================================================

# !pip install -q langchain langchain-community chromadb
# !pip install -q sentence-transformers pypdf tiktoken
# !pip install -q google-generativeai gradio

# =====================================================================
# CELL 2 - imports
# =====================================================================

import os
import re
import warnings
import urllib.request
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

# langchain handles the RAG orchestration
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# gemini as the LLM (free tier, fast)
import google.generativeai as genai

# gradio for the chat UI
import gradio as gr

print("imports done")

# =====================================================================
# CELL 3 - configuration
# =====================================================================

# paste your free Gemini API key here
# get one at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# the 7 papers and where to download them from
PAPERS = {
    "Attention Is All You Need": {
        "file": "attention.pdf",
        "url":  "https://arxiv.org/pdf/1706.03762.pdf"
    },
    "BERT": {
        "file": "bert.pdf",
        "url":  "https://arxiv.org/pdf/1810.04805.pdf"
    },
    "GPT-3": {
        "file": "gpt3.pdf",
        "url":  "https://arxiv.org/pdf/2005.14165.pdf"
    },
    "RAG": {
        "file": "rag.pdf",
        "url":  "https://arxiv.org/pdf/2005.11401.pdf"
    },
    "Sentence-BERT": {
        "file": "sbert.pdf",
        "url":  "https://arxiv.org/pdf/1908.10084.pdf"
    },
    "LoRA": {
        "file": "lora.pdf",
        "url":  "https://arxiv.org/pdf/2106.09685.pdf"
    },
    "Llama 2": {
        "file": "llama2.pdf",
        "url":  "https://arxiv.org/pdf/2307.09288.pdf"
    }
}

# chunk settings
# I tried a few values here - 1200 chars with 200 overlap worked well
# smaller chunks = more precise retrieval but lose surrounding context
# overlap makes sure nothing important falls exactly on a boundary
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
TOP_K         = 5    # how many chunks to retrieve per query

DB_FOLDER = "./chroma_db"

# =====================================================================
# CELL 4 - download papers
# =====================================================================

# arxiv blocks automated downloads so I set a browser-like user agent
# this took me a while to figure out - without it you get a 403 error

def download_all_papers():
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; research-student)'}

    for name, info in PAPERS.items():
        fpath = info["file"]
        if Path(fpath).exists():
            print(f"  already have: {fpath}")
            continue

        print(f"  downloading {name}...")
        try:
            req = urllib.request.Request(info["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(fpath, 'wb') as f:
                    f.write(resp.read())
            print(f"  done: {fpath}")
        except Exception as e:
            print(f"  failed ({e}) - download manually from {info['url']}")

download_all_papers()

# =====================================================================
# CELL 5 - load and chunk the papers
# =====================================================================

# RecursiveCharacterTextSplitter tries to split on paragraph breaks first
# then sentences, then words - keeps semantic units together better
# than just cutting at fixed character positions

def load_and_chunk():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        separators      = ["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for paper_name, info in PAPERS.items():
        fpath = info["file"]
        if not Path(fpath).exists():
            print(f"  skipping {paper_name} - file not found")
            continue

        print(f"  loading {paper_name}...")
        loader = PyPDFLoader(fpath)
        pages  = loader.load()

        # tag each page with the paper name so we can cite it later
        for page in pages:
            page.metadata['paper'] = paper_name

        chunks = splitter.split_documents(pages)

        # light cleanup - remove page numbers and excessive whitespace
        # arxiv PDFs have a lot of header/footer noise
        for chunk in chunks:
            text = chunk.page_content
            text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
            text = re.sub(r' {2,}', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            chunk.page_content = text.strip()

        all_chunks.extend(chunks)
        print(f"    {len(pages)} pages -> {len(chunks)} chunks")

    print(f"\ntotal chunks: {len(all_chunks)}")
    return all_chunks


chunks = load_and_chunk()

# =====================================================================
# CELL 6 - create embeddings and store in ChromaDB
# =====================================================================

# sentence-transformers/all-MiniLM-L6-v2 is a lightweight but good
# embedding model - converts text to 384-dimensional vectors
# similar meaning = similar vectors = shows up in search results
#
# ChromaDB is a simple local vector database
# it stores chunks + their embedding vectors so we can do fast similarity search
# persist_directory saves it to disk so we don't need to re-embed every time

print("loading embedding model...")
embedder = HuggingFaceEmbeddings(
    model_name   = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {'device': 'cpu'}
)
print("embedding model ready")

print(f"\nembedding {len(chunks)} chunks into ChromaDB...")
print("this takes 2-3 minutes the first time, cached after that...")

vector_db = Chroma.from_documents(
    documents         = chunks,
    embedding         = embedder,
    persist_directory = DB_FOLDER,
    collection_name   = "research_papers"
)
vector_db.persist()

print("vector store ready and saved to disk")

# =====================================================================
# CELL 7 - setup Gemini LLM
# =====================================================================

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')

def ask_llm(prompt):
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"LLM error: {str(e)}"

# quick test
print(ask_llm("reply with just the word hello"))

# =====================================================================
# CELL 8 - the RAG function
# =====================================================================

# this is the core of the whole thing
# query comes in -> embed it -> find similar chunks in DB
# -> build a prompt with those chunks as context -> LLM answers
#
# the prompt template tells the LLM:
#   1. only answer from the context given
#   2. mention which paper the answer came from
#   3. admit if you don't have enough info (reduces hallucination)
#
# without instruction 3 the LLM will just make something up
# which defeats the whole point of RAG

PROMPT = """You are a research assistant with access to NLP research papers.
Use ONLY the context below to answer the question.
Always mention which paper your answer comes from.
If the context doesn't have enough info, say: "The provided papers don't cover this sufficiently."

Context:
{context}

Question: {question}

Answer:"""


def rag_answer(question, top_k=TOP_K):
    # step 1: find the most relevant chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    relevant  = retriever.get_relevant_documents(question)

    # step 2: build the context string from retrieved chunks
    context_parts = []
    sources = []

    for i, doc in enumerate(relevant):
        paper_name = doc.metadata.get('paper', 'Unknown')
        page_num   = doc.metadata.get('page', '?')
        context_parts.append(f"[Source {i+1}: {paper_name}, p.{page_num}]\n{doc.page_content}")
        if paper_name not in sources:
            sources.append(paper_name)

    context = "\n\n---\n\n".join(context_parts)

    # step 3: fill in the prompt and send to LLM
    full_prompt = PROMPT.format(context=context, question=question)
    answer      = ask_llm(full_prompt)

    return answer, sources, relevant


# =====================================================================
# CELL 9 - test with the example queries from the task
# =====================================================================

test_questions = [
    "How does self-attention differ from recurrence?",
    "What problem does RAG solve?",
    "How does LoRA reduce training cost?",
    "What is the difference between BERT and GPT?",
    "How does Sentence-BERT create sentence embeddings?",
    "What is the role of the feed-forward network in a transformer?",
]

print("=" * 65)
for q in test_questions:
    ans, srcs, _ = rag_answer(q)
    print(f"\nQ: {q}")
    print(f"Sources used: {', '.join(srcs)}")
    print(f"A: {ans[:350]}...")
    print("-" * 65)

# =====================================================================
# CELL 10 - multi-paper comparison query
# =====================================================================

# showing the system can pull from multiple papers at once
# this is the more interesting use case

comparison_q = "Compare how BERT, GPT-3, and LoRA approach the problem of training large language models efficiently. What does each paper contribute?"

ans, srcs, chunks_used = rag_answer(comparison_q, top_k=8)

print("Multi-paper comparison:")
print("=" * 65)
print(ans)
print(f"\nPapers referenced: {srcs}")

# =====================================================================
# CELL 11 - Gradio chat interface
# =====================================================================

# building a proper interactive UI
# Gradio is the simplest way to get a chat UI in Colab
# share=True gives a public URL that works for the screen recording

def chat(message, history):
    if not message.strip():
        return "Ask me something about the research papers!"

    answer, sources, _ = rag_answer(message)

    source_line = "\n\n📄 **Sources:** " + " | ".join(sources)
    return answer + source_line


demo = gr.ChatInterface(
    fn          = chat,
    title       = "Spider Research Paper Assistant",
    description = (
        "Ask anything about: Transformers, BERT, GPT-3, RAG, "
        "Sentence-BERT, LoRA, or Llama 2.\n"
        "Answers are retrieved from the actual papers, not from memory."
    ),
    examples = [
        "How does self-attention work?",
        "What problem does RAG solve?",
        "What is LoRA and how does it reduce parameters?",
        "Compare BERT and GPT architectures",
        "How does Llama 2 handle safety?",
        "What are the limitations of the original transformer?",
    ],
    theme = gr.themes.Soft(),
)

# share=True creates a public URL - use this for your screen recording
demo.launch(share=True, debug=False)

# =====================================================================
# CELL 12 - retrieval analysis
# =====================================================================

# checking which papers get retrieved most for different types of queries
# useful to verify the embedding model is routing queries correctly

check_queries = [
    "self-attention mechanism query key value",
    "fine-tuning pretrained language models",
    "sentence similarity cosine distance",
    "low rank matrix decomposition weights",
    "knowledge retrieval from external documents",
    "bidirectional context representation",
    "instruction following chat model",
]

print("Retrieval routing check:")
print("-" * 55)

for q in check_queries:
    docs    = vector_db.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(q)
    top_src = Counter(d.metadata.get('paper', '?') for d in docs).most_common(1)[0][0]
    print(f"  '{q[:42]:<42}' → {top_src}")
