# ReAct Agent
# A React Agent is a specific type of agent that can "reason". Practically, this means looping between tools and llm output
# until some exit condition, usually relating to answer quality, is met. This looping is named "reasoning" or "thinking"
# The purpose of this exercise is to create a robust react agent. To achieve this we will:
#   1. Learn how to create Tools in LangGraph
#   2. Learn how to create a ReAct Graph
#   3. Introduce new types of Messages called ToolMessage, BaseMessage and SystemMessage
#   4. Test robustness of the state machine

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Define the agent state. Notice now we define the state with general super types
# Also add_messages reducer function is will provide built in support for recording
# chat history as we go
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Notice we wrote a detailed docstring in standard python form. This is required
# so that our state machine, guided by our LLM, can recognise what tool to use
# for different operations it wants to achieve
@tool
def add(a: int, b: int) -> int:
    """
    This is an addition function that adds 2 numbers together
    
    :param a: First number
    :type a: int
    :param b: Second number
    :type b: int
    :return: Result of addition of first and second numbers
    :rtype: int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """
    This is a subtraction function that substracts second number from the first
    
    :param a: First number
    :type a: int
    :param b: Second number
    :type b: int
    :return: Result of subtraction
    :rtype: int
    """
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """
    This is a multiplication function that multiplies 2 numbers
    
    :param a: First number
    :type a: int
    :param b: Second number
    :type b: int
    :return: Result of multiplication of first and second numbers
    :rtype: int
    """
    return a * b

# Include all tools in a list
tools = [add, multiply, subtract]
# And bundle them with your LLM model
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    """
    This function calls the LLM model
    """
    # System prompt below is the natural language instructions the developer gives to the model
    system_prompt = SystemMessage(
        content="You are my AI assistant, answer my query to the best of your ability."
    )
    # Notice the system prompt is appened to the state['messages']
    # This is because this prompt doesn't necessarily represent a state.
    # It's a constant to this node. So we can just append it at any position
    # beginning, end anywhere in between depending on our use case in this.
    response = model.invoke([system_prompt] + state['messages'])
    # This is how you append the messages via the reducer function add_messages defined above
    return {'messages': [response]}

def should_continue(state: AgentState) -> str:
    """
    This function implements the halting condition of our state machine.
    """
    # We check if we still need to call any tools. If not, we exit
    if not state['messages'][-1].tool_calls:
        return "exit"
    else:
        return "continue"

# Lastly, we define this special kind of node that contains all the tools we have
# Our model can lookup this node, and pick a tool to use. 
tool_node = ToolNode(tools=tools)

# As always, define and compile the graph
graph = StateGraph(AgentState)
graph.add_node(model_call.__name__, model_call)
graph.add_node(tool_node.name, tool_node)
graph.add_edge(START, model_call.__name__)
graph.add_conditional_edges(
    model_call.__name__,
    should_continue,
    {
        "exit": END,
        "continue": tool_node.name
    }
)
graph.add_edge(tool_node.name, model_call.__name__)
app = graph.compile()

# For visualisation when need be
# from io import BytesIO
# from PIL import Image
# image = Image.open(BytesIO(app.get_graph().draw_mermaid_png()))
# image.show()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 3 + 2 and then multiply the result by 6")]}
print_stream(app.stream(inputs, stream_mode="values"))