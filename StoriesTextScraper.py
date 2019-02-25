import requests
import sqlite3
import DBEngine
from sqlalchemy import create_engine, MetaData, Table
import multiprocessing
import sys
from bs4 import BeautifulSoup
from sqlalchemy.orm import sessionmaker

class StoriesTextScraper(multiprocessing.Process):

    def __init__(self, start=0, skip=12):
        multiprocessing.Process.__init__(self)
        self.starting = start
        self.skip = skip

    def run(self):
        engine = create_engine(DBEngine.database_location)
        metadata = MetaData()
        Session = sessionmaker(bind=engine)
        session = Session()
        metadata.reflect(engine)
        story_page_table = Table(DBEngine.StoryPageTable, metadata)
        story_text_table = Table(DBEngine.StoryTextsTable, metadata)
        i=self.starting
        for column in session.query(story_page_table).yield_per(1000):
            try:
                if i % self.skip==0:
                    id, title, url=column
                    print(id)
                    if url is None: 
                        continue
                    request = requests.get(url).text.encode(sys.stdout.encoding, errors='replace')
                    content = BeautifulSoup(request, "lxml").find_all('p',text=True)
                    for c in content:
                        session.execute(story_text_table.insert(), {"text":c.get_text(), "id_source":id})
                        session.commit()
                i+=1
            except Exception as e: 
                print(e)