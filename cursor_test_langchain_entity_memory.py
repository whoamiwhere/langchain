from langchain.llms import OpenAI
from langchain.memory import ConversationEntityMemory

from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict, Any

import pandas as pd
import os

os.environ['OPENAI_API_KEY'] = "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"
llm = OpenAI(temperature=0)

memory = ConversationEntityMemory(llm=llm)
_input = {"input": "Deven & Sam are working on a hackathon project"}
_output = {"output": " That sounds like a great project! What kind of project are they working on?"}
memory.save_context(
    _input,_output 
)

_input = {"input": "They are trying to add more complex memory structures to Langchain"}
_output = {"output": " That sounds like an interesting project! What kind of memory structures are they trying to add?"}
memory.save_context(
    _input,_output 
)

_input = {"input": " They are adding in a key-value store for entities mentioned so far in the conversation."}
_output = {"output": " That sounds like a great idea! How will the key-value store help with the project?"}
memory.save_context(
    _input,_output 
)

_input = {"input": " Sam is the founder of a company called Daimon."}
_output = {"output": "That's impressive! It sounds like Sam is a very successful entrepreneur. What kind of company is Daimon?"}
memory.save_context(
    _input,_output 
)

conversation = ConversationChain(
    llm=llm, 
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=memory
)
answer = conversation.predict(input="What do you know about Deven & Sam?")
print(answer)
print(conversation.memory.entity_store.store)