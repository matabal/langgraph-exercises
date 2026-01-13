# Drafter Agent
# Imagine a company with the problem of spending too much time drafting writtent communication documents. These can be internal memos, emails, manuscripts etc.
# Your task as an AI Engineer is to create an AI agent that will work with the user to draft documents fast.
# This task needs to be done in a Human-AI feedback loop. Meaning that AI Agent won't halt until the user is happy with the draft.
# The system should be able to save drafts.

from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Global variable to store document content
# Note: Lookup InjectorState in LangGraph for how this is done in a production environment
document_content = ""

# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# We will have two tools: update and save
@tool
def update(content: str) -> str:
    """Updates the current document content with provided content"""
    global document_content
    document_content = content
    return f"Document updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """
    Saves the current document to a text file 
    
    :param filename: Name of the text file to save into.
    :type filename: str
    :return: Explanation of the action taken and success state
    :rtype: str
    """
    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return "saved"
    
    except Exception as e:
        return f"error saving"

tools = [update, save]

# Initialise the model
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview").bind_tools(tools)

# Define the nodes
def writer(state: AgentState) -> AgentState:
    """This node writes the document content based on user instructions"""
    system_prompt = SystemMessage(content=f"""
    You are a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use 'update' tool with the complete updated content.
    - If the user wants to save and finish the, use the 'save' tool.
    - Make sure to always show the current document state after modifications. 
    """)

    if not state['messages']:
        user_input = input("What would you like to write today?")
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    response = model.invoke(all_messages)

    # To print all tool calls. Can comment out if need be
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return { 'messages': list(state['messages']) + [user_message, response] }

tool_node = ToolNode(tools)

# Define halting condition
def should_continue(state: AgentState) -> str:
    """Determine if the execution should continue"""

    messages = state['messages']
    if not messages:
        return "continue"

    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.content == "saved":
            return "end"
    return "continue"

# Create the graph
graph = StateGraph(AgentState)
graph.add_node(writer.__name__, writer)
graph.add_node(tool_node.name, tool_node)
graph.add_edge(START, writer.__name__)
graph.add_edge(writer.__name__, tool_node.name)
graph.add_conditional_edges(
    tool_node.name,
    should_continue,
    {
        "continue": writer.__name__,
        "end": END
    }
)
app = graph.compile()

# For visualisation when need be
# from io import BytesIO
# from PIL import Image
# image = Image.open(BytesIO(app.get_graph().draw_mermaid_png()))
# image.show()

# Function to print message stream
def print_messages(messages: Sequence[BaseMessage]):
    """Function to print message stream"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

# Main function
def run_drafter_agent():
    print("\n ===== DRAFTER =====")

    initial_state = { 'messages': [] }
    for step in app.stream(initial_state, stream_mode="values"):
        if "messages" in step:
            print_messages(step['messages'])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_drafter_agent()