# Chatbot
# In this exercise, we will create a chatbot that can remember message history so it can have a context when chatting with the user.
# Hence the main goal is to introduce a form of memory for our agent. 
# To achieve this we will:
#   1. Use different message types - HumanMessage and AIMessage
#   2. Maintain a full conversation history using both message types
#   3. Use Gemini model with LangChain built-in libraries
#   4. Create a sophiesticated conversation loop

from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# Define the state. 
# The key here is we are keeping HumanMessages and AIMessages in order in a single list
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# And the model we'll use
llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

# Define the nodes
def ask(state: AgentState) -> AgentState:
    """
    This node collects a question from the user.
    """
    user_input = input("User=> ")
    state['messages'].append(HumanMessage(content=user_input))
    return state

def answer(state: AgentState) -> AgentState:
    """
    This node produces a response based on the last user message and chat history
    """
    response = llm.invoke(state['messages'])
    print(f"AI=> {response.text}")
    state['messages'].append(AIMessage(content=response.text))
    return state

# Define exit logic
def should_continue(state: AgentState) -> str:
    """
    This function determines when the application should halt
    """
    if state['messages'][-1].content != "exit":
        return "continue"
    else:
        return "exit"
    
# Define the graph and compile
graph = StateGraph(AgentState)
graph.add_node(ask.__name__, ask)
graph.add_node(answer.__name__, answer)
graph.add_edge(START, ask.__name__)
graph.add_conditional_edges(
    ask.__name__,
    should_continue,
    {
        "continue": answer.__name__,
        "exit": END
    }
)
graph.add_edge(answer.__name__, ask.__name__)
app = graph.compile()

# For visualisation when need be
# from io import BytesIO
# from PIL import Image
# image = Image.open(BytesIO(app.get_graph().draw_mermaid_png()))
# image.show()

# Run the graph
initial_state = { "messages": [] }
final_state = app.invoke(initial_state)

# Additionally, if you want your chat history to be saved. You can save it in a document
# and read it everytime application restarts. A more sophisticated way of doing this is 
# through Vector Databases. Which we'll get to.