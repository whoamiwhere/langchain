import os 
os.environ["OPENAI_API_KEY"] = "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"
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains.question_answering import load_qa_chain

# load document
loader = PyPDFLoader("C:/Users/28440/Desktop/论文/GPT/tutorials-LangChain-main/materials/example.pdf")
documents = loader.load()

### For multiple documents 
# loaders = [....]
# documents = []
# for loader in loaders:
#     documents.extend(loader.load())


# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# select which embeddings we want to use
embeddings = OpenAIEmbeddings()
# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
# create a chain to answer questions 
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=False)
query = "what is the total number of AI publications in 2010?"
print(qa.run(query))


