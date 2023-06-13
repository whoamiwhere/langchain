from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

db_user = "SELECTmember1"
db_password = "123456"
db_host = "localhost"
db_name = "classicmodels"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
     
# llm=OpenAI(temperature=0)
import os
from langchain.chat_models import ChatOpenAI
os.environ['OPENAI_API_KEY'] = "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"
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
# chat = ChatOpenAI(model_name="gpt-4")
     

toolkit = SQLDatabaseToolkit(db=db)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# agent_executor.run("Describe the Order related table and how they are related")
# agent_executor.run("Find the top 5 products with the highest total sales revenue")
# agent_executor.run("List top 3  countries with the highest number of orders")
agent_executor.run(" Which country's customers spent the most?List the total sales and the country.")