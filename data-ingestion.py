## warning filter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## importing dependencies
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

## loading the data
documentRaw = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader).load()

## splitting the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documentSplit = splitter.split_documents(documents=documentRaw)

## creating embeddings & storing into a knowledge-base
embedF = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
vectorDB = FAISS.from_documents(embedding=embedF, documents=documentSplit)

## saving the knowledge-base locally
vectorDB.save_local("vector-store")

print("Succesfully saved!!")