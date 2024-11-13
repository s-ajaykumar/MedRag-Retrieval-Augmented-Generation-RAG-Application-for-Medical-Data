#Import libraries
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import json
import base64
from io import BytesIO
from uuid import uuid4
from blob_authentication import blob_client
from DocumentLoader import CustomDocumentLoader
from postgres_server import postgres_client
from langchain_core.messages import AIMessage, HumanMessage
from PostgresChatMessageHistory_class import PostgresChatMessageHistory
from client_inputs_saver import client_inputs
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from langchain_chroma import Chroma
from langserve import add_routes
from fastapi import FastAPI
from langserve import CustomUserType
from pydantic import Field


class FileProcessingRequest(CustomUserType):
    """Request including a base64 encoded file."""

    # The extra field is used to specify a widget for the playground UI.
    file: list = Field(...)
    file_name: list = Field(...)
    
    
    
#Callback for collecting the llm response and push to the database
class LoggingHandler(BaseCallbackHandler):
  def on_llm_end(self, response: LLMResult, **kwargs) -> None:
      rag_obj.push_chat_history(HumanMessage(content = client_inputs.input_dic['prompt']), AIMessage(content = response.generations[0][0].message.content))
callbacks = [LoggingHandler()]    
      
      
      
class rag:
        def __init__(self, blob_client, postgres_client):
            self.blob_client = blob_client
            self.postgres_client = postgres_client
            self.postgres_conn = postgres_client.sync_connection
            #Load the embedder and llm
            self.embedder = OllamaEmbeddings(model = "mxbai-embed-large")
            self.llm = ChatOllama(model = "llama3.2")
            #Load the vector store
            #Download from azure blob storage and load it
            self.vector_store = Chroma(embedding_function = self.embedder, persist_directory = "/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/Vector_Store_Chroma")
            #initialize the retriever
            self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
            #Prompt structure
            self.prompt = PromptTemplate.from_template("""Chat history: {chat_history}
            user question: {question}
            context: {context}
            prompt: Imagine youself as a person who converses with your fellow doctor.
            Check whether the context is relevant to the user question and if It's relevant then respond to the user 
            question from the context else respond from you own knowledge.
            The response should be in a natural conversational manner.
            Only respond to the question. 
            Simplify the response.""")

        #Concatenate retrieved passages
        def format_docs(self, docs):
                return "\n\n".join(doc.page_content for doc in docs)
          
        def upload_blob_data(self, container_name: str, file_object, file_name_string):
            container_client = self.blob_client.get_container_client(container=container_name)
            # Upload the blob data - default blob type is BlockBlob
            container_client.upload_blob(name = file_name_string, data = file_object, overwrite = True)

        def _process_file(self, request: FileProcessingRequest) -> str:
            
            file_name_strings = []

            for name in request.file_name:
                file_name_bytes_data = base64.b64decode(name.encode("utf-8"))
                file_name_strings.append(file_name_bytes_data.decode('utf-8'))

            file_num = 0
            docs = []

            for req in request.file:
                content = base64.b64decode(req.encode("utf-8"))
                file_object = BytesIO(content)
                self.upload_blob_data("rawdocstore", file_object, file_name_strings[file_num])
                file_num += 1
                loader = CustomDocumentLoader(file_object)
                docs.extend(doc for doc in loader.load())
            uuids = [str(uuid4()) for _ in range(len(docs))]
            self.vector_store.add_documents(documents = docs, ids = uuids)
            #Reinitialize retriever
            self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
            return "File/Files uploaded and vector_store updated"

        def list_postgres_tables(self):
            cursor = self.postgres_conn.cursor()
            cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_schema = 'public'
            """)
            # Fetch the table names
            table_names = []
            tables = cursor.fetchall()
            for table in tables:
                table_names.append(table[0])
            # Close the cursor and connection
            cursor.close()
            return table_names
            
        def get_chat_history(self, user_id, session_id):
          if user_id not in self.list_postgres_tables():
            self.postgres_client.add_table(user_id)          ##Need to check whether instead of postgres_client, self.postgres_client should be replaced.
            self.chat_history = PostgresChatMessageHistory(
            user_id,
            session_id,
            sync_connection=self.postgres_conn
            )
            return "No chat history. New user."
          else:
            table_name = user_id 
            chat_history_string = ''
            self.chat_history = PostgresChatMessageHistory(
            table_name,
            session_id,
            sync_connection=self.postgres_conn
            )
            for chat in self.chat_history.messages:
              chat_history_string += f"{chat.type}: {chat.content}" + '\n'
            return chat_history_string
          
        def push_chat_history(self, prompt, response):
          self.chat_history.add_messages([prompt, response])       
rag_obj = rag(blob_client, postgres_client)
  


#Chain the components
rag_chain = (
    RunnableLambda(lambda x: client_inputs.inputs(x))
    | RunnableMap(
      {"chat_history" : lambda inputs: rag_obj.get_chat_history(inputs['user_id'], inputs['session_id']),
       "context": RunnableLambda(lambda inputs: inputs['prompt']) | rag_obj.retriever | rag_obj.format_docs,
       "question": lambda inputs: inputs['prompt']}
    )
    | rag_obj.prompt
    | rag_obj.llm
    | StrOutputParser()
)
rag_chain_with_callback = rag_chain.with_config(callbacks = callbacks)


#FastAPI
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, rag_chain_with_callback, path = "/stream", enable_feedback_endpoint=True)
add_routes(app, 
           RunnableLambda(rag_obj._process_file).with_types(input_type=FileProcessingRequest),
           config_keys=["configurable"],
           path="/file_upload",)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)



