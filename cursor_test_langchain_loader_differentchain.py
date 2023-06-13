from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
import os

os.environ['OPENAI_API_KEY'] = "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"

with open('C:/Users/28440/Desktop/论文/GPT/tutorials-LangChain-main/state_of_the_union.txt', encoding='utf-8') as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()

query = "What did the president say about Justice Breyer"
docs = docsearch.get_relevant_documents(query)

# -------------------------------------------------------------
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# --------------------------1.The stuff Chain (two using method)
# chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
# a = chain.run(input_documents=docs, question=query)

# -----------------------2.The map_reduce Chain
# chain = load_qa_chain(OpenAI(batch_size=5,temperature=0), chain_type="map_reduce", return_map_steps=True)

# ----------------------3.The refine Chain
# chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine", return_refine_steps=True)

# ----------------------4.The map-rerank Chain
chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_rerank", return_intermediate_steps=True)

a = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
print(a)