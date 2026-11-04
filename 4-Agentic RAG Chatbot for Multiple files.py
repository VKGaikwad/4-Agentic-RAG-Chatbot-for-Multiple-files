from langchain_community.document_loaders import TextLoader, PyPDFLoader
from google.colab import files
import os


uploaded = files.upload()

all_docs = []

for filename in uploaded.keys():
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext == ".pdf":
        print(f"📄 {filename} → PDF detected")
        loader = PyPDFLoader(filename)

    elif file_ext == ".txt":
        print(f"📄 {filename} → Text detected")
        loader = TextLoader(filename)

    else:
        print(f"⚠️ Skipping unsupported file: {filename}")
        continue

    docs = loader.load()

    # ✅ attach filename metadata (VERY IMPORTANT for RAG)
    for d in docs:
        d.metadata["source"] = filename

    all_docs.extend(docs)

print(f"\n✅ Loaded {len(all_docs)} total document pages")


###############

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(all_docs)

print(f"✅ Generated {len(chunks)} chunks")

###############

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

################

import uuid
from langchain_community.vectorstores import Chroma

# Generate a unique directory for each run
persist_path = f"/content/chroma_db_{uuid.uuid4()}"

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_path
)
db.persist()
retriever = db.as_retriever()

###############

from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
import os

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Set the GROQ_API_KEY environment variable")

llm = ChatOpenAI(
    model_name="llama-3.3-70b-versatile",
    api_key=api_key,
    openai_api_base="https://api.groq.com/openai/v1",  # Important!
    temperature=0.3
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

for c in chunks[:7]:
    print(c.metadata["source"])
    
query = "What are these documents about?"
response = qa_chain.run(query)
print("Answer:", response)
