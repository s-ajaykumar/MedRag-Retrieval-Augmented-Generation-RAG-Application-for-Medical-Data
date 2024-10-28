import streamlit as st
from langserve import RemoteRunnable
import base64


stream_runnable = RemoteRunnable("http://localhost:8000/stream")
file_upload_runnable = RemoteRunnable("http://localhost:8000/file_upload")
#runnable = RemoteRunnable("https://99e6-106-205-74-176.ngrok-free.app/word/")

def response(prompt, stream_runnable):
    chunks = []
    with st.chat_message("Assistant"):
        for chunk in stream_runnable.stream(prompt):
            chunks.append(chunk)
            yield chunk
    st.session_state['full_response'] = ''.join(chunks)

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
        
    

#If the prompt is not None
with st.sidebar:
    files = st.file_uploader(label = "Upload Files here", accept_multiple_files = True, label_visibility = 'collapsed')
    #st.button("send", on_click = send_files(files))

if len(files) > 0:
    st.status(label = "running", expanded = False, state = 'running')
    send_files(files)

st.chat_message("Assistant")
st.markdown("Hi, I am MedGPT, your beloved medical document assistant. Feel free to ask anything:)")


#Save and load chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

else:
    for message in st.session_state.messages:
        with st.chat_message('user'):
            st.markdown(message)
        with st.chat_message('Assistant'):
            st.markdown(st.session_state['full_response'])


if prompt := st.chat_input("Type here"):
    

    with st.chat_message("User"):
        st.markdown(prompt)
    response = st.write((response(prompt, stream_runnable)))
    st.session_state.messages.append(prompt)




    

