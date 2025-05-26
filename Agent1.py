from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17",
                            google_api_key=os.environ['GOOGLE_API_KEY'])

def process(state:AgentState)->AgentState:
    response = llm.invoke(state['messages'])
    print(f"\nAI:{response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()


user_input = input("Enter: ")
while user_input != 'bye':
    agent.invoke({"messages":[HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
