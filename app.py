## set up environment
## set up environment
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")

## langchain dependencies
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain.load import loads, dumps

## other dependencies
from typing import List
import time


## supress langchain warning
import warnings
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

## setting up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")
persistent_directory = os.path.join(current_dir, "data-ingestion-local")

## instantiate embedding model
embedF = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

## load local vector DB
vectorDB = Chroma(embedding_function=embedF, persist_directory=persistent_directory)

## set up retriever
kb_retriever = vectorDB.as_retriever(search_type="similarity",search_kwargs={"k": 5})

## main RAG prompt template
with open("prompts/mainRAG-prompt.md") as f:
    mainRAGPrompt = f.read()

## multiquery schema
class MultiQuery(BaseModel):
    documentRetrievalRequired: bool = Field(
        default=False, 
        description="Flag indicating whether the user query requires document retrieval from the knowledge base. Set to False for conversational queries like greetings, confirmations, or chitchat that don't need external information."
    )
    generatedQueries: List[str] = Field(
        default=[], 
        description="List of semantically equivalent query variations for improved retrieval coverage. Empty list if documentRetrievalRequired is False. Contains only the original query if it's straightforward, or 5 alternative phrasings if the query is complex/ambiguous."
    )

def createMultiQueryChain(output_pydantic_object, llm):
    # llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.15)
    with open("prompts/multiQuery-prompt.md", "r") as f:
        template = f.read()
    parser = JsonOutputParser(pydantic_object=output_pydantic_object)
    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "user_query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    return chain

def generateRRF(all_docs: List[List[Document]], k: int = 60) -> List[Document]:
    rrf_scores = dict()
    for docs in all_docs:
        for rank, doc in enumerate(docs, start=1):
            rrf_rank = 1/(k+rank)
            rrf_scores[dumps(doc)] = rrf_scores.get(dumps(doc), 0) + rrf_rank
    sorted_rrf_score = sorted(rrf_scores.items(), key=(lambda x: x[1]), reverse=True)
    best_docs = [loads(doc) for doc, _ in sorted_rrf_score][:3] ## only top 3 documents
    return best_docs

def generateResponse(
        chat_history: List, 
        llm,
        documents: List[Document] = None,
        main_RAG_template=mainRAGPrompt
) -> str:
    if documents is None:
        context = ""
    else:
        context = "\n"
        for rank, doc in enumerate(documents, start=1):
            context += f"Relevant Document: {rank}" + doc.page_content + "\n"
    prompt = ChatPromptTemplate.from_template(main_RAG_template)
    chain = prompt | llm | StrOutputParser()
    resp = chain.invoke({
        "chat_history": chat_history,
        "context": context
    })
    return resp

if __name__=="__main__":
    llm = ChatCohere(temperature=0.15)

    ## initial chat state
    chat_history = []
    welcome_message = "Welcome to the Legal Assistant Bot. How can I help you today? Write `exit` to quit."
    chat_history.append(AIMessage(content=welcome_message))
    
    ## print welcome message
    print("\033[1m> AI:\033[0m ", end="")
    for letter in welcome_message:
        print(letter, flush=True, end="")
        time.sleep(0.03)
    

    ### ----------------------------------------------------------------------------------------------------
    
    ## retrieved docs
    
    while True:
        try:
            retrieved_docs = []

            ## user input
            print("\n\033[1m> Human:\033[0m ", end="")
            user_query = input().strip()
            
            ## check if user wants to exit
            if user_query.lower().strip() == "exit":
                bye_message = "Chat ended! See ya."
                print("\033[1m> AI:\033[0m ", end="")
                for letter in bye_message:
                    print(letter, flush=True, end="")
                    time.sleep(0.03)
                break

            ## multiquery
            chain = createMultiQueryChain(output_pydantic_object=MultiQuery, llm=llm)
            resp = chain.invoke({"chat_history": chat_history, "user_query": user_query})
            chat_history.append(HumanMessage(content=user_query))

            if resp['documentRetrievalRequired']:
                ## retrieve documents for each query
                multiQueries = resp['generatedQueries']
                for query in multiQueries:
                    retrieved_docs.append(kb_retriever.invoke(query))
                
                ranked_documents = generateRRF(retrieved_docs)
                ai_resp = generateResponse(chat_history=chat_history, llm=llm, documents=ranked_documents)
            else:
                ai_resp = generateResponse(chat_history=chat_history, llm=llm)
            
            ## print AI response
            print("\033[1m> AI:\033[0m ", end="")
            for letter in ai_resp:
                print(letter, flush=True, end="")
                time.sleep(0.03)
            
            ## append AI response to chat history
            chat_history.append(AIMessage(content=ai_resp))
        except KeyboardInterrupt:
            print("\n\n[ALERT] ^C: Keyboard Interruption detected. Session Terminated!!")
            break