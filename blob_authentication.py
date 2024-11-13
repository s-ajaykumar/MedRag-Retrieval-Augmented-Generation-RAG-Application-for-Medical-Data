from azure.storage.blob import BlobServiceClient

class blob_authentication:
   #Authentication
    def __init__(self):
      self.account_url = "https://medaidevstore.blob.core.windows.net"
      self.credential = "CIgfLjFLblENeeXYNANxY8BTDtp69XDGtC4HLLy5fItZ3BUQ8qh3kA6LhjUBwNI9jH15RTBV5URw+AStEkKkAQ=="
      # Create the BlobServiceClient object
      self.blob_service_client = BlobServiceClient(self.account_url, credential=self.credential)
blob_obj = blob_authentication()
blob_client = blob_obj.blob_service_client