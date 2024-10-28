#pip install langchain_ollama 
#Import libraries
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
#from langchain_core.messages import AIMessage
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langserve import add_routes
from fastapi import FastAPI


#Load the embedder and llm
embedder = OllamaEmbeddings(model = "mxbai-embed-large")
llm = ChatOllama(model = "llama3.2")

#Load the vector store
#Download from azure blob storage and load it
vector_store = Chroma(embedding_function = embedder, persist_directory = "/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/New_chromaDB")

#initialize the retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

#Prompt structure
prompt = hub.pull("rlm/rag-prompt")

#Concatenate retrieved passages
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#Chain the components
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is pain"):
    print(chunk, end = "", flush = True)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, rag_chain, path = "/stream", enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)



