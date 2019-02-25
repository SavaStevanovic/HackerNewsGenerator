import requests
import json
import sqlite3
import os.path
from pathlib import Path
import DBEngine
from sqlalchemy import create_engine, MetaData, Table
import multiprocessing
import sys


class StoriesScraper(multiprocessing.Process):

    def __init__(self, start, skip=12):
        multiprocessing.Process.__init__(self)
        self.linkBase = "https://hacker-news.firebaseio.com/v0/item/%d.json"
        self.starting = start
        self.skip = skip

    def run(self):
        engine = create_engine(DBEngine.database_location)
        metadata = MetaData(bind=engine)
        story_page_table = Table(DBEngine.StoryPageTable, metadata, autoload=True)

        end_counter = 0
        i = self.starting
        while end_counter < 1000:
            try:
                i += self.skip
                request = requests.get(self.linkBase % i).text.encode(sys.stdout.encoding, errors='replace')
                content = json.loads(request)
                if content is None:
                    end_counter += 1
                    continue
                end_counter = 0
                if content["type"] == "story":
                    print(content["title"])
                    engine.execute(story_page_table.insert(), id=content["id"], title=content["title"], url=content["url"])
            except:
                end_counter += 1
