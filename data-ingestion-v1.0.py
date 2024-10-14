import time
## settings up the env
import os
from dotenv import load_dotenv
load_dotenv()

## langchain dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

## setting up directories
current_dir_path = os.path.dirname(os.path.abspath(__file__)) ## <- extracting the directory name from the absolute path of this file
data_path = os.path.join(current_dir_path, "data") ## creating a path for the `data` folder
persistent_directory = os.path.join(current_dir_path, "data-ingestion-local") ## creating a directory to save the vector store locally

## checking if the directory already exists
if not os.path.exists(persistent_directory):
    print("[INFO] Initiating the build of Vector Database .. 📌📌", end="\n\n")

    ## checking if the folder that contains the required PDFs exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"[ALERT] {data_path} doesn't exist. ⚠️⚠️"
        )

    ## list of all the PDFs
    pdfs = [pdf for pdf in os.listdir(data_path) if pdf.endswith(".pdf")] ## <- making a list of all file names as str

    doc_container = [] ## <- list of chunked documents aka container
    
    ## taking each each item from `pdfs` and loading it using PyPDFLoader
    for pdf in pdfs:
        loader = PyPDFLoader(file_path=os.path.join("data", pdf),
                             extract_images=False)
        docsRaw = loader.load() ## <- returns a list of `Document` objects. Each such object has - 1. Page Content // 2. Metadata
        for doc in docsRaw:
            doc_container.append(doc) ## <- appending each `Document` object to the previously declared container (list)

    ## splitting the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs_split = splitter.split_documents(documents=doc_container)

    ## displaying information about the split documents
    print("\n--- Document Chunks Information ---", end="\n")
    print(f"Number of document chunks: {len(docs_split)}", end="\n\n")

    ## embedding and vector store
    embedF = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2") ## <- open-source embedding model from HuggingFace - taking the default model only
    print("[INFO] Started embedding", end="\n")
    start = time.time() ## <- noting the starting time

    """
    creating the embeddings for the documents and
    then storing to a vector database
    """
    vectorDB = Chroma.from_documents(documents=docs_split,
                                     embedding=embedF,
                                     persist_directory=persistent_directory)
    
    end = time.time() ## <- noting the end time
    print("[INFO] Finished embedding", end="\n")
    print(f"[ADD. INFO] Time taken: {end - start}")

else:
    print("[ALERT] Vector Database already exist. ️⚠️")