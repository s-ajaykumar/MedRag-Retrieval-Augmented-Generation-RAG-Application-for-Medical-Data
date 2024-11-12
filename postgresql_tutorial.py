import psycopg2
conn = psycopg2.connect(database = "testdb", host = "localhost", port = 5432)
cursor = conn.cursor()

#Fetch data
'''cursor.execute("""SELECT * FROM ajay;
               """)
for row in cursor.fetchall():
    print(row)'''
    
#Create tables and insert data
'''cursor.execute("""CREATE TABLE IF NOT EXISTS test_table(id int);
               """)
cursor.execute("INSERT INTO test_table VALUES(003)")'''

# Fetch and print table names
'''cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables
    WHERE table_schema = 'public'
""")
tables = cursor.fetchall()
for table in tables:
    print(table[0])'''

#Drop a table
'''cursor.execute("DROP TABLE ajay2;")'''
conn.commit()
cursor.close()
conn.close()