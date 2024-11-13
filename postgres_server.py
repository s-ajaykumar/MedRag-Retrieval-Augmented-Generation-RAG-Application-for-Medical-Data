import psycopg2
from PostgresChatMessageHistory_class import PostgresChatMessageHistory

class postgres_server:
    def __init__(self):
        # (or use psycopg.AsyncConnection for async)
        self.sync_connection = psycopg2.connect(database = "testdb", host = "localhost", port = 5432)

    def add_table(self, user_id):
        # Create the table schema (only needs to be done once)
        PostgresChatMessageHistory.create_tables(self.sync_connection, user_id)
        #session_id = str(uuid.uuid4())
postgres_client = postgres_server() 