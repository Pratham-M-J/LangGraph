from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import os

load_dotenv()

#global variable to pass in a state in tools, (injected state)
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_document(content:str)->str:
    """This function updates the document with the provided content"""
    global document_content
    document_content = content
    return f"Document updated successfully!: {document_content}"

@tool
def save_document(filename: str)->str:
    """This function saves the document to a file
    Args:
        filename (str): The name of the file to save the text(.txt) file to.
    """
    if not filename.endswith('.txt'):
        filename += f"{filename}.txt"
    
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\nDocument saved successfully to {filename}")
        return f"Document saved successfully to {filename}"
    except Exception as e:
        print(f"\nError saving document: {e}")
        return f"Error saving document: {e}"
    
tools = [update_document, save_document]

llm = ChatOpenAI(model="gpt-4o-mini",
                 openai_api_key=os.environ['OPEN_AI_KEY']).bind_tools(tools)
                             
def our_agent(state:AgentState)->AgentState:
    system_message = SystemMessage(
        content = f"""
        You are a Drafter, a heplpful writing assistant, You are gonna help the user update and modify documents.
        - If the user wants to update or modify content use the 'update_document' tool with the complete updated content.
        - If the user wants to save the document, use the 'save_document' tool with the filename.
        - Make sure ti always show the current document state after modification.

        The current document content is: {document_content}
    """
    )
    if not state['messages']:
        user_input = "Im ready to help you update a document, What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do? ")
        print(f"\nUser: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_message] + list(state['messages']) + [user_message]
    response = llm.invoke(all_messages)
    print(f"\nAI: {response.content}")
    return {"messages": list(state["messages"])+ [user_message, response]}

