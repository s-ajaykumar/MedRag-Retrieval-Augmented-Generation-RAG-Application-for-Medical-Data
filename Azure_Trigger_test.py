from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobClient
from io import BytesIO
import docx
import os

import logging
import config
import azure.functions as func
import azurefunctions.extensions.bindings.blob as blob


#Create blob_service_client
def get_blob_service_client_account_key():
    
    account_url = config.blob_url
    shared_access_key = config.blob_key
    credential = shared_access_key

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=credential)

    return blob_service_client

blob_service_client = get_blob_service_client_account_key()



def get_blob_client(blob_service_client: BlobServiceClient, container_name, blob_name):
    # Create a blob client using the service client object
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return blob_client

blob_client = get_blob_client(blob_service_client, "rawdocstore", "sample-blob.docx")


def update_blob_file(blob_service_client: BlobServiceClient, container_name: str):
    container_client = blob_service_client.get_container_client(container=container_name)
    with open(file=os.path.join('/Users/outbell/Ajay/DeepLearning/GenAI/MedLLM/VSCode/temp', 'sample-blob.docx'), mode="rb") as data:
        blob_client = container_client.upload_blob(name="sample-blob.docx", data=data, overwrite=True)
    print("File updated successfully")

update_blob_file(blob_service_client, "rawdocstore")


