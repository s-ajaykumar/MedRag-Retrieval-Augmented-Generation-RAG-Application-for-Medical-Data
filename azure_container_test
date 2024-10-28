from azure.storage.blob import BlobServiceClient
import os
import config


#Authentication
def get_blob_service_client_account_key():
    
    account_url = config.blob_url
    shared_access_key = config.blob_key
    credential = shared_access_key

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=credential)

    return blob_service_client

blob_service_client = get_blob_service_client_account_key()


#Download blobs
'''def download_blob_to_file(blob_service_client: BlobServiceClient, container_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob="sample-blob.docx")
    with open(file=os.path.join(r'/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/VSCode/temp/temp', 'sample-blob.docx'), mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(download_stream.readall())

download_blob_to_file(blob_service_client, "rawdocstore")'''


#Delete blobs
'''def delete_blob(blob_service_client: BlobServiceClient, container_name: str, blob_name: str):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.delete_blob()

delete_blob(blob_service_client, "rawdocstore", "sample-blob.txt")'''


#Upload blobs
'''def upload_blob_file(blob_service_client: BlobServiceClient, container_name: str):
    container_client = blob_service_client.get_container_client(container=container_name)
    with open(file=os.path.join('/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/VSCode/server_test_data', 'Analgesic fix copy.docx'), mode="rb") as data:
        blob_client = container_client.upload_blob(name="sample-blob.docx", data=data, overwrite=True)
    print("File uploaded successfully")

upload_blob_file(blob_service_client, "rawdocstore")'''


#List blobs
'''def list_blobs_flat(blob_service_client: BlobServiceClient, container_name):
    container_client = blob_service_client.get_container_client(container=container_name)

    blob_list = container_client.list_blobs()

    for blob in blob_list:
        print(f"Name: {blob.name}")

list_blobs_flat(blob_service_client, "rawdocstore")'''


#List containers
"""def list_containers(blob_service_client: BlobServiceClient):
    containers = blob_service_client.list_containers(include_metadata=True)
    for container in containers:
        print(container['name'], container['metadata'])

list_containers(blob_service_client)"""



