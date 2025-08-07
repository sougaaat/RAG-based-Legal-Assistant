## set up environment
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")

## langchain dependencies
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, AIMessage

## other dependencies
from typing import List

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

def createMultiQueryChain(output_pydantic_object):
    # llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.15)
    llm = ChatCohere(temperature=0.15)
    with open("../prompts/multiQuery-prompt.md", "r") as f:
        template = f.read()
    parser = JsonOutputParser(pydantic_object=output_pydantic_object)
    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "user_query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm | parser
    return chain

## dummy input <> output
if __name__ == "__main__":
    dummy_chat_history = [
        HumanMessage(content="I'm working on a research project about renewable energy technologies."),
        AIMessage(content="That sounds interesting! What specific aspects of renewable energy are you focusing on?"),
        HumanMessage(content="I'm particularly interested in solar and wind power efficiency improvements over the last decade."),
        AIMessage(content="Great focus area! Solar panel efficiency has improved significantly, and wind turbine technology has also advanced considerably. Are you looking at any specific metrics or geographical regions?"),
        HumanMessage(content="Mainly focusing on technological breakthroughs rather than regional data."),
        AIMessage(content="Sounds great. Tell me what you want to know.")
    ]

    user_query = "btw I wanted to say that I'm so happy to talk to you."
    chain = createMultiQueryChain(output_pydantic_object=MultiQuery)
    
    ## generate output
    resp = dict()
    count = 0
    while 'generatedQueries' not in resp:
        if count == 3:
            raise Exception("Output is not parsable!")
        resp = chain.invoke({"chat_history": dummy_chat_history, "user_query": user_query})
        count += 1
    
    ## display output
    print(resp)
    print(f"\n\n> Number of MultiQueries: {len(resp["generatedQueries"])}")