from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain import SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain

db_user = "root"
db_password = "123456"
db_host = "localhost"
db_name = "classicmodels"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
     
# llm=OpenAI(temperature=0)
import os
os.environ['OPENAI_API_KEY'] = "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"
llm = OpenAI(temperature=0)
# chat = ChatOpenAI(model_name="gpt-4")

from langchain.prompts.prompt import PromptTemplate

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

If someone asks for the table foobar, they really mean the employee table.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

# db_chain = SQLDatabaseChain(llm=llm, database=db, prompt=PROMPT, verbose=True)
# db_chain.run("Which customers spent more money top 10? Please provide a list of some of those customers and their spending amounts.")

chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True)
chain.run("Which customers spent more money top 10? Please provide a list of some of those customers and their spending amounts.")