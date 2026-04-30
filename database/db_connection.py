import psycopg
import os
from dotenv import load_dotenv

class DBConnector:
    def __init__(self, dotenv_path):
        load_dotenv(dotenv_path)
        envs = ['PG_DB_PASSWORD', 'PG_DB_USER', 'PG_DB_DATABASE', 'PG_DB_HOST', 'PG_DB_PORT']
        for env in envs:
            if not os.environ.get(env):
                raise Exception(f'Environment variable {env} is not set')

        self.conn = psycopg.connect(
            dbname=os.getenv('PG_DB_DATABASE'),
            user=os.getenv('PG_DB_USER'),
            password=os.getenv('PG_DB_PASSWORD'),
            host=os.getenv('PG_DB_HOST'),
            port=os.getenv('PG_DB_PORT'),
            sslmode="require"
        )

        


    def close_connection(self):
        if self.conn:
            self.conn.close()
    def query(self, query, args=None):
        with self.conn.cursor() as cursor:
            cursor.execute(query, args)
            return cursor.fetchall()

    def execute_query(self, query, params=None):
        if not self.conn:
            raise Exception("Database connection is not established.")
        
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
    
    def add_data(self, data):
        if not self.conn:
            raise Exception("Database connection is not established.")
        query = """INSERT INTO books (title, description, embedding) 
        VALUES (%(title)s, %(description)s, %(embedding_vector)s) ON CONFLICT (title) DO NOTHING;"""
        try:
            for item in data:
                params ={
                    'title': item['book_name'],
                    'description': item['summaries'],
                    'embedding_vector': item['embedding']
                }       
                self.execute_query(query, params)
                print(f"Data inserted successfully for title: {item['book_name']}")
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting data: {e}")
        