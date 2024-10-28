import base64
from langserve import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/word/")
#runnable = RemoteRunnable("https://99e6-106-205-74-176.ngrok-free.app/word/")

file_paths = ["/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/VSCode/server_test_data/Analgesic fix copy.docx"]

encoded_data = []
for file_path in file_paths:
    with open(file_path, "rb") as f:
        data = f.read()
    encoded_data.append(base64.b64encode(data).decode("utf-8"))

print(runnable.invoke({"file": encoded_data}))













"""with open("/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/VSCode/Analgesic fix copy 3.docx", "rb") as f:
    data = f.read()

encoded_data = base64.b64encode(data).decode("utf-8")


requests.post(
    "http://localhost:8000/word/invoke", json={"input": {"file": encoded_data}}
).json()"""