from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import json
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict

import pandas as pd
import os

os.environ['OPENAI_API_KEY'] = "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"

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")
# memory.save_context({"input": "hi"}, {"output": "whats up"})
print(type(memory))

# memory.load_memory_variables({})

llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=memory # ConversationBufferMemory()
)

# conversation 1
# input = "Hi there!"
# answer  = conversation.predict(input=input)
# print(answer)

# history = ChatMessageHistory()
# history.add_user_message(input)
# history.add_ai_message(answer)
# dicts = messages_to_dict(history.messages)
# new_messages = messages_from_dict(dicts)


# conversation 2

input="I'm doing well! Just having a conversation with an AI."
answer = conversation.predict(input = input)
print(answer)