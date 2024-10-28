#pip install faiss-cpu langchain_community uuid
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from io import BytesIO


#To treat binary string as file object:
'''
#Read as binary string
with open("/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/VSCode/server_test_data/Analgesic fix copy.docx", 'rb') as f:
    file = f.read()

#Read as file object
file_object = BytesIO(file) '''



#Load the existing FAISS vector store
'''db = FAISS.load_local(
    "faiss_index", embedder, allow_dangerous_deserialization=True
)

#Index the documents and store in the vector store
uuids = [str(uuid4()) for _ in range(len(docs))]
db.add_documents(documents=docs, ids=uuids)

#Replace the new faiss vector store with the existing vectorstore in local
db.save_local("faiss_index")'''