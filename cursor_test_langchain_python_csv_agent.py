from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import pandas as pd
import os
os.environ['OPENAI_API_KEY'] = "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"

agent = create_csv_agent(OpenAI(temperature=0), 'C:/Users/28440/Desktop/data/titanic.csv', verbose=True)
# agent.run("how many rows are there?")
# agent.run("how many people have more than 3 siblings")
# agent.run("whats the square root of the average age?")
agent.run("calculate survival rate of different cabin class")