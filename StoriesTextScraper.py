import requests
import sqlite3
import DBEngine
from sqlalchemy import create_engine, MetaData, Table
import multiprocessing
import sys
from bs4 import BeautifulSoup
from sqlalchemy.orm import sessionmaker


class StoriesTextScraper(multiprocessing.Process):

    def __init__(self, column, lock):
        multiprocessing.Process.__init__(self)
        self.column = column
        self.lock = lock

    def run(self):
        engine = create_engine(DBEngine.database_location)
        metadata = MetaData(bind=engine)
        metadata.reflect(engine)
        story_text_table = Table(DBEngine.StoryTextsTable, metadata)
        try:
            if self.column.url is None:
                return
            request = requests.get(self.column.url).text.encode(sys.stdout.encoding, errors='replace')
            content = BeautifulSoup(request, "lxml").find_all('p', text=True)
            for c in content:
                with self.lock:
                    with engine.connect() as connection:
                        connection.execute(story_text_table.insert().values(text=c.get_text(), id_source=self.column.id))
        except Exception as e:
            print(e)
