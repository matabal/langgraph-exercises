# Retrieval-Augmented Generation (RAG) Agent for Ancient Life the Mediterranean
# 
# In this exercise, we will build a RAG Agent to researching information about
# ancient life in the Mediterranean basin based on famous historian Braudel's work.
# 
# Braudel's work is known to be quite detailed and deep so it can be tedious to quickly
# search and summarise his writings. Nonetheless his work is considered as the cornerstone of
# modern study of Mediterranean history and in fact the modern practice of all history.
# 
# Thus the purpose of this agent to help a social science researches that often uses Braudel's work
# in his own research.
# 
# The PDF that will constitute the knowledge base of this agent is taken from the completely open
# source Internet Archive from the following url: 
# https://ia800807.us.archive.org/18/items/TheMediterraneanInTheAncientWorld/The_Mediterranean_in_the_Ancient_World.pdf


from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from operator import add as add_messages
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

# temperature= controls the randomness and creativity of the model. Technically this means how
# probability distribution should be when selecting the next token (word or character)
# In this instant we have 0 as we just want the model to summarise what it reads and not take
# any risks. 
llm = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    temperature=0
)

# Embedding Models
# We need to utilise embedding models to create efficient and cost-effective RAG Agents.
# This is because we simply have too much data to load to the LLM in a single go.
# In this exercise for instance 485 pages. But this can be easily 10K+ pages in a more
# comprehentive agent.
# 
# So embedding models essentially convert your text to multi-dimensional semantic vector
# space. This basically means they encode each word as a vector and most importantly 
# vectors (words) with similar meanings are **closer** to each other.
#
# For instance, in a simple string match search, words "car" and "automobile" are two
# seperate entities. However an embedding model have a context to closely relate "car"
# and "automobile"
# 
# These embeddings will later on be loaded to a Vector Database (Chrome in this exercise)
# and will be made persistent across usage sessions.

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

pdf_path = "The_Mediterranean_in_the_Ancient_World.pdf"
pdf_loader = PyPDFLoader(pdf_path)

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
# Below is the chucking process we apply to our knowledge base text before sending it
# to our Vector DB. We want to parse the information in chunks because of its immense
# size. 
# 
# RecursiveCharacterTextSplitter achieves this by assuming text that are physically close
# to each other are semantically related.
# 
# It's parameters are:
# chunk_size: The maximum number of characters (or tokens) you want in a single chunk (eg. 1000)
# chunk_overlap: The number of characters shared between the end of one chunk and the start of the next (eg. 200)
# Overlap is crucial because it prevents information loss at the "seams."
# If a sentence is cut in half between two chunks, the overlap ensures the full thought is preserved 
# in at least one of them.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Complete chunking
pages_split = text_splitter.split_documents(pages)

# Configure variables for the persistence layer (Chroma)
# Note: make sure to have PERSIST_DIRECTORT env var in your .env file
persist_dir = os.getenv("PERSIST_DIRECTORT")
collection_name = "The_Mediterranean_in_the_Ancient_World"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

# Initiliase Chroma DB
try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise e

# Retriever
# In this section, we are creating a retriever object using chroma db, and 
# convert it into a tool that our LLM can interact with.

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tools searches and returns information from The_Mediterranean_in_the_Ancient_World document.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in The_Mediterranean_in_the_Ancient_World document."
    
    # We need to return the results as a single coherent text
    results =[]
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

# Finally register and bind our tools
tools = [retriever_tool]
llm = llm.bind_tools(tools)

# Now for the State Machine
# We will build a similar state machine to the ReAct Agent's state machine we built before.
# Effectively this will mean the state machine will loop between the llm and the retriever
# until it reaches to some meaningful halting condition.

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Our halting condition for this exercise is a single answer.
# In other words, if we looked up our db once and llm provided an answer using
# the information retrieved, we stop.
def should_continue(state: AgentState) -> str:
    """Checks if the last message was a tool call"""
    result = state['messages'][-1]
    if hasattr(result, 'tool_calls') and len(result.tool_calls) > 0:
        return "continue"
    else:
        return "exit"

# Define our system prompt
system_prompt = """
You are an intelligent AI assistant for a researcher studying ancient Mediterranean history. Your job is to assist the researcher with
Fernand Braudel's work on the subject matter. You may find all Fernand Braudel's work on this subject matter in your knowledge base. Spefically
your knowledge base contains Fernand Braudel's The Mediterranean in the Ancient World book. When the user asks a question on the subject matter,
use your retriever tool to lookup the content of this book and answer their question accordingly. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that.
Always cite the specific parts of the documents you use in your answers.
"""

# Lookup to use later for llm tool call validation
tools_dict = {our_tool.name: our_tool for our_tool in tools}

# Nodes
# LLM Agent Node
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    response = llm.invoke(messages)
    return { 'messages': [response] }

# Retriever Node
def call_retriever(state: AgentState) -> AgentState:
    """Executes tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        # For verbosity
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        # Checks if a valid tool is present
        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))

        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    
    print("Tools Execution Complete. Back to the model!")
    return { 'messages': results }

# Configure the Graph
graph = StateGraph(AgentState)
graph.add_node(call_llm.__name__, call_llm)
graph.add_node(call_retriever.__name__, call_retriever)
graph.add_edge(START, call_llm.__name__)
graph.add_conditional_edges(
    call_llm.__name__,
    should_continue,
    {
        "exit": END,
        "continue": call_retriever.__name__
    }
)
graph.add_edge(call_retriever.__name__, call_llm.__name__)
app = graph.compile()

# For visualisation when need be
# from io import BytesIO
# from PIL import Image
# image = Image.open(BytesIO(app.get_graph().draw_mermaid_png()))
# image.show()

def run_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = app.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].text)


run_agent()

