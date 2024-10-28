
from langserve import RemoteRunnable


runnable = RemoteRunnable("http://localhost:8000/stream")
#runnable = RemoteRunnable("https://99e6-106-205-74-176.ngrok-free.app/word/")


for chunk in runnable.stream("I have more pain"):
    print(chunk, end = "", flush = True)













"""with open("/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/VSCode/Analgesic fix copy 3.docx", "rb") as f:
    data = f.read()

encoded_data = base64.b64encode(data).decode("utf-8")


requests.post(
    "http://localhost:8000/word/invoke", json={"input": {"file": encoded_data}}
).json()"""