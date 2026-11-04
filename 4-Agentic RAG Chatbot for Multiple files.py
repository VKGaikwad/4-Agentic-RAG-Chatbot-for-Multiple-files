import numpy
import numpy
import streamlit as st
print("NUMPY VERSION:", numpy.__version__)
print("✅ SETUP DONE — RESTART RUNTIME NOW")

# 📌 1. Upload Your File

uploaded_files = st.sidebar.file_uploader(
    "Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True
)

# Ingest documents
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("/content", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Choose loader
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            loader = Docx2txtLoader(file_path)

        docs = loader.load()


        # 🔧 Step 2: Split the Document into Chunks
        # NOTE - For effective RAG, give size of chunks and overlap based on your document size

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(docs)

        print(f"✅ Generated {len(chunks)} chunks")

        # 🔑 Step 3: Create Embeddings
        # You can use OpenAI, HuggingFace, or Ollama (if local setup). We'll be proceeding with HuggingFace.

        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


        # 🧠 Step 4: Store Chunks in Chroma Vector DB


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


        # 🧠 Step 5: Ask Questions About Your Document
        # Now that your document is embedded and stored in a vector database, you can ask natural language questions about it.

        # We'll use a free LLM via the Groq API, which runs powerful models like llama3-8b at extremely fast speeds.

        # ⚠️ To use Groq, you need an API key.
        # Please refer to the "Obtaining Groq API Key" file in the Lab 0 - Setup folder for instructions on how to get and configure your key.

        from langchain_openai import ChatOpenAI
        from langchain_classic.chains import RetrievalQA
        import os

        api_key = os.environ.get("GROQ_API_KEY")`r`n        if not api_key:`r`n            raise RuntimeError("Set the GROQ_API_KEY environment variable")`r`n        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(
            model_name="llama-3.3-70b-versatile",
            openai_api_base="https://api.groq.com/openai/v1",  # Important!
            temperature=0.3
        )

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        for c in chunks[:7]:
            print(c.metadata["source"])
            
        query = "What are these documents about?"
        response = qa_chain.run(query)
        print("Answer:", response)
