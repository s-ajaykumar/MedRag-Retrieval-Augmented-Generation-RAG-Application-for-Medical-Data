from langchain_huggingface import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(model_name = "/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/stella_model_file/",
                                   model_kwargs={'device': 'cpu'})