from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
import os

os.environ['OPENAI_API_KEY'] = "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"
# os.environ['SERPAPI_API_KEY'] ="6aefcd80-d777-11ed-919b-4b36835706c9"
# 4.only 999 left 
os.environ['SERPER_API_KEY'] = "ab6d842051b8bae7a969625b9678bf0cb62cc3c3"

# search = GoogleSerperAPIWrapper()

llm = OpenAI(temperature=0)
tools = load_tools(["google-serper", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("How many Teslas have been sold in 2022. Multiple by 2")