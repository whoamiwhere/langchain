import os
import langchain
from langchain.llms import OpenAI, Cohere, HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

os.environ['OPENAI_API_KEY'] = "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"
# chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo')
# gpt3 = OpenAI(model_name='text-davinci-003')
# cohere = Cohere(model='command-xlarge')
# flan = HuggingFaceHub(repo_id="google/flan-t5-xl")


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

from langchain.document_loaders import TextLoader
loader = TextLoader('C:/Users/28440/Desktop/论文/GPT/tutorials-LangChain-main/state_of_the_union.txt', encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Simplied Chinese:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(),
 chain_type_kwargs=chain_type_kwargs)

query = "What did the president say about Ketanji Brown Jackson"
print(qa.run(query))
