import os
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = "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"

# 1.per day 100
os.environ["GOOGLE_CSE_ID"] = "e32b4dc353dc94bfd"
os.environ["GOOGLE_API_KEY"] = "AIzaSyARk_qakfGrGweu49xCZOPh9X8Zo7q27Y8"

# search = GoogleSearchAPIWrapper()

# 3.per month 100
os.environ["SERPAPI_API_KEY"] = "0407f140b2b974a77c883e2c260f527b25ac9780eb8ab65c27d0f1ba44521414"

# 2.bing per month 1000
os.environ["BING_SUBSCRIPTION_KEY"] = "79e9aab3ce31407c90509e25d2d6f485" # '2268c101efd34f6c877dff867002f76c'
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

# 4.only 999 left 
os.environ['SERPER_API_KEY'] = "ab6d842051b8bae7a969625b9678bf0cb62cc3c3"



llm = OpenAI(temperature=0)
tools = load_tools(["google-search", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("How many Teslas have been sold in 2022. Multiple by 2")