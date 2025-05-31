from langchain_core.tools import Tool
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_experimental.utilities import PythonREPL
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=os.environ['GOOGLE_API_KEY']
)

# Create a PythonREPL instance
# This provides an environment where Python code can be executed as strings
python_repl = PythonREPL()

# Create a Tool using the Tool class
# This wraps the Python REPL functionality as a tool that can be used by agents
python_calculator = Tool(
    # The name of the tool - this helps agents identify when to use this tool
    name="Python Calculator",
    
    # The function that will be called when the tool is used
    # python_repl.run takes a string of Python code and executes it
    func=python_repl.run,
    
    # A description of what the tool does and how to use it
    # This helps the agent understand when and how to use this tool
    description="Useful for when you need to perform calculations or execute Python code. Input should be valid Python code."
)

tools = [python_calculator]

template = """
You are a helpful calculator agent. You can perform basic arithmetic operations.

AVAILABLE TOOLS:
----------------
You have access to the following tools: {tool_names}
Here are the details for each tool:
{tools}

INSTRUCTIONS FOR USING TOOLS:
-----------------------------
To use a tool, you MUST use the following format.
First, think about which tool you need from the list of available tools: {tool_names}.
Then, specify the chosen tool name in the "Action" line.
The "Action Input" line MUST be a single, valid JSON string. This JSON string should represent a dictionary where keys are the argument names of the tool (as described in the tool details above) and values are the actual data for those arguments.

Example of using the 'addition' tool:
Thought: I need to add 5 and 3. The 'addition' tool is suitable. Its arguments are 'a' and 'b'.
Action: addition
Action Input: {{"a": 5, "b": 3}}

After you use a tool, the system will provide an "Observation":
Observation: [The result of the tool action, or an error message if it failed.]

You will then continue with a new "Thought", and potentially another "Action" or a "Final Answer".
Your thoughts should reflect your reasoning process.

If a tool produces an error (e.g., "ValueError: Division by zero is not allowed."), this error will be in the "Observation".
You should analyze the error in your "Thought" and decide how to proceed (e.g., inform the user, try a different approach).

If you believe you have the answer to the user's question, you MUST use the following format for your final response:
Thought: I have now arrived at the final answer based on my calculations and observations.
Final Answer: [Your comprehensive final answer to the original question]

Begin!

Question: {input}
{agent_scratchpad}
"""

# Define the input variables list explicitly
# THIS IS THE MOST IMPORTANT PART TO GET RIGHT FOR THIS ERROR
prompt_input_variables_list = ["input", "tools", "tool_names", "agent_scratchpad"]

prompt = PromptTemplate(
    input_variables=prompt_input_variables_list, # Use the defined list
    template=template
)

# --- !! ADD THIS DEBUGGING PRINT STATEMENT !! ---
print("----------------------------------------------------")
print(f"DEBUG: Type of prompt object: {type(prompt)}")
print(f"DEBUG: Prompt input_variables: {prompt.input_variables}")
print(f"DEBUG: Is 'tool_names' in prompt.input_variables? {'tool_names' in prompt.input_variables}")
print("----------------------------------------------------")

# This is line 116 in your traceback
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

queries = [
    "solve integration of log(x^3+e^x^2) where x=2, then multiply it by 34, also tell me a joke about it",
]

for user_input in queries:
    print(f"\nUser: {user_input}")
    try:
        response = agent_executor.invoke({"input": user_input})
        print(f"AI: {response['output']}")
    except Exception as e:
        print(f"An unexpected error occurred during agent execution: {e}")
        # If the error is from create_react_agent, it might not even reach here.
    print("-" * 50)