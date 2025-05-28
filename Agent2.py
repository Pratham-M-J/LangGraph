from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]] #In langchain and langgraph HumanMessage and AIMessage are the datatypes
    #doing this we are able to store both Human and AI messages in the very list defined

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17",
                            google_api_key=os.environ['GOOGLE_API_KEY'])

def process(state:AgentState)->AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI:{response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter:")
while user_input!='bye':
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages":conversation_history})
    conversation_history = result["messages"]
    print(conversation_history)
    user_input = input("Enter: ")

with open("logging.txt", "w") as file:
    file.write("Your conversation log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("End of convo")

