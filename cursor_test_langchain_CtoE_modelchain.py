from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate
import os

os.environ['OPENAI_API_KEY'] = "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"

first_prompt_template = """
As a professional translator, I need you to accurately and efficiently translate the following Chinese text into English while preserving the meaning of the original context.
The Chinese text is as follows:
{Chinese_content}

"""

human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=first_prompt_template,
            input_variables=["Chinese_content"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chat = ChatOpenAI(temperature=0.2)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)
# print(chain.run("请介绍一下机器学习中的KNN并给出他在一些情景下的使用案例-需详细说明"))
second_prompt = PromptTemplate(
    input_variables=["English_question"],
    template="{English_question},and translate the answer into simplied chinese",
)
chain_two = LLMChain(llm=chat, prompt=second_prompt)
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
English_answer = overall_chain.run("请介绍一下机器学习中的KNN并给出他在一些情景下的使用案例仅4-5句子即可")
print(English_answer)