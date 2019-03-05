import requests
import sqlite3
import DBEngine
from sqlalchemy import create_engine, MetaData, Table
import multiprocessing
import sys
from bs4 import BeautifulSoup
from sqlalchemy.orm import sessionmaker


class StoriesTextScraper(multiprocessing.Process):
    lock=None
    def __init__(self, column):
        multiprocessing.Process.__init__(self)
        self.column = column

    def run(self):
        with self.lock:
            engine = create_engine(DBEngine.database_location)
            metadata = MetaData(bind=engine)
            metadata.reflect(engine)
            story_text_table = Table(DBEngine.StoryTextsTable, metadata)
        try:
            if self.column.url is None:
                return
            request = requests.get(self.column.url).text.encode(sys.stdout.encoding, errors='replace')
            content = BeautifulSoup(request, "lxml").find_all('p', text=True)
            with self.lock:
                for c in content:
                    with engine.connect() as connection:
                        connection.execute(story_text_table.insert().values(text=c.get_text(), id_source=self.column.id))
        except Exception as e:
            pass
