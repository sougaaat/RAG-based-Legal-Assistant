import os
from dotenv import load_dotenv
load_dotenv()

os.environ['USER_AGENT'] = 'myagent'
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")

import logging
logging.getLogger('sagemaker').setLevel(logging.ERROR)

# Load docs
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Store splits
vectorstore = FAISS.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings(model_name = "shibing624/text2vec-base-multilingual"))
print("[INFO] Finished building KB.", end="\n")
# LLM
llm = ChatCohere()

condense_question_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

condense_question_prompt = ChatPromptTemplate.from_template(condense_question_template)

qa_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. Use three sentences maximum and keep the
answer concise.

Chat History:
{chat_history}

Other context:
{context}

Question: {question}
"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)

convo_qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    condense_question_prompt=condense_question_prompt,
    combine_docs_chain_kwargs={
        "prompt": qa_prompt,
    },
)
from operator import itemgetter
getter = itemgetter('question', 'answer')
result = convo_qa_chain.invoke(
    {
        "question": "What are autonomous agents?",
        "chat_history": "",
    }
)

print(getter(result)[0], "\n", getter(result)[1])