import streamlit as st
from langserve import RemoteRunnable
import base64

stream_runnable = RemoteRunnable("http://localhost:8000/stream")
file_upload_runnable = RemoteRunnable("http://localhost:8000/file_upload")
#runnable = RemoteRunnable("https://99e6-106-205-74-176.ngrok-free.app/word/")
    
def response(stream_runnable, input):
    return stream_runnable.stream(input)

def send_files(files):
    file_content_encoded_data = []
    file_name_encoded_data = []
    for file in files:
        #convert file_name string into bytes and then into base64 
        file_name_bytes_data = file.name.encode('utf-8')
        file_content_bytes_data = file.read() 
        file_name_encoded_data.append(base64.b64encode(file_name_bytes_data).decode('utf-8')) 
        file_content_encoded_data.append(base64.b64encode(file_content_bytes_data).decode('utf-8'))
    #if file_upload_runnable.invoke({'file' : file_content_encoded_data, 'file_name' : file_name_encoded_data}) == :
        
def ui():
    #If the prompt is not None
    with st.sidebar:
        files = st.file_uploader(label = "Upload Files here", accept_multiple_files = True, label_visibility = 'collapsed')
        #st.button("send", on_click = send_files(files))
    st.chat_message("Assistant")
    st.markdown("Hi, I am MedGPT, your beloved medical document assistant. Feel free to ask anything:)")
ui()
    
#User input
user_id = 'ajay'
session_id = 'ef6173ca-89ac-4130-987e-6b6cb9c5e8e9'
if prompt := st.chat_input("Type here"):
    with st.chat_message("User"):
        st.markdown(prompt)
    response = st.write(response(stream_runnable, {"prompt" : prompt, "user_id" : user_id, "session_id" : session_id}))




    

