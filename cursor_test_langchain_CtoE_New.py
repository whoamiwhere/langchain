from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate
import os

os.environ['OPENAI_API_KEY'] = "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"



template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chat = ChatOpenAI(temperature=0)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)
# print(chain.run("请介绍一下机器学习中的KNN并给出他在一些情景下的使用案例仅4-5句子即可"))


English_question = chain.run(input_language="Chinese", 
output_language="English",text= """我爱学习""")
print(English_question)