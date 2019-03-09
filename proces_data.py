import DBEngine
import multiprocessing
from StoriesScraper import StoriesScraper
from StoriesTextScraper import StoriesTextScraper
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
import re
import os
if __name__ == '__main__':
    DBEngine.initiate_database()
    engine = create_engine(DBEngine.database_location)
    Session = sessionmaker(bind=engine)
    session = Session()
    i = 0
    f = open('document.txt', 'a', encoding="utf-8")
    s = ''
    for c in session.query(DBEngine.StoryText).yield_per(1000):
        if len(c.text) > 30:
            if c.id_source == i:
                s += c.text+' eop '
            elif s != '':
                s = s.replace('\n', ' eol ')
                s = re.sub('\s+', ' ', s).strip()
                s += s[:-7] + ' eot'+ os.linesep
                f.writelines(s.lower())
                s = c.text+' eop '
                print(c.id_source)
            i = c.id_source
