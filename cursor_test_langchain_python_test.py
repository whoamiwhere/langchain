from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import os
os.environ['OPENAI_API_KEY'] = "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"

df = pd.read_csv(f"C:/Users/28440/Desktop/data/titanic.csv")
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

# agent.run("how many rows are there?")
# agent.run("how many people have more than 3 siblings")
# agent.run("whats the square root of the average age?")
agent.run("calculate survival rate of different cabin class")