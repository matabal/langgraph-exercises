# For this exercise, convert the agent in simple_boy.ipynb to a proper chat bot where the agent and the user can chat without the execution being interrupted
# Choosing a standard python file for this one as it's easier to collect user input

from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# Define the state
class AgentState(TypedDict):
    messages: List[HumanMessage]

# And the model we'll use
llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

# Define nodes
def ask(state: AgentState) -> AgentState:
    """
    This node collects a question from the user.
    """
    user_input = input("User=> ")
    state['messages'] = [HumanMessage(content=user_input)]
    return state

def answer(state: AgentState) -> AgentState:
    """
    This node produces a response based on the user given message
    """
    response = llm.invoke(state['messages'])
    print(f"AI=> {response.text}")
    return state

# Define should continue logic
def should_continue(state: AgentState) -> str:
    if (state['messages'][0].content != "exit"):
        return "continue"
    else:
        return "exit"

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

initial_state = { "messages": [] }
final_state = app.invoke(initial_state)
