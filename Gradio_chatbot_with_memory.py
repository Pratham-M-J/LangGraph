from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
import gradio as gr
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",  
    google_api_key=os.environ.get('GOOGLE_API_KEY'), 
  
)


chat_template = """
You are a very friendly chatbot, Alice.
Think thrice before answering, don't hallucinate. If possible, provide sources.
Answer the user's questions with grace.

Before answering, say:
'''
🤔Thinking...
'''

Once you have the answer, format like this:

Answer: <your answer here>


Current conversation:
{chat_history}

User's Question:
{query}
"""

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=False
)


prompt = PromptTemplate(
    input_variables=["chat_history", "query"],  
    template=chat_template
)


chat_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False  
)


def invoke_chat_chain(user_query: str):
    """
    Invokes the LLMChain with the user's query.
    The chain automatically handles history.
    """
    response = chat_chain.invoke({"query": user_query})
    return response.get('text', "Sorry, I couldn't process that.")




chat_app = gr.Interface(
    fn=invoke_chat_chain,  # Use the corrected function
    inputs=gr.Textbox(label="Your Question", lines=3, placeholder="Type your query here..."),
    outputs=gr.Textbox(label="Alice's Reply"),
    title="Stellar.AI - Chat with Alice",
    description="Yo!!! I'm Alice, your friendly AI companion. Ask me anything!"
)


if __name__ == "__main__":
    chat_app.launch(server_name="127.0.0.1", server_port=7860)