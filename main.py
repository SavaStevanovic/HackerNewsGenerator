import DBEngine
import multiprocessing
from StoriesScraper import StoriesScraper
from StoriesTextScraper import StoriesTextScraper
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

scrapeUrls = False
scrapeText = True
lock = multiprocessing.Lock()
if __name__ == '__main__':
    DBEngine.initiate_database()
    cpu_count = multiprocessing.cpu_count()
    if scrapeUrls:
        processes = [StoriesScraper(i, cpu_count) for i in range(cpu_count)]
        for p in processes:
            p.start()
    if scrapeText:
        try:
            engine = create_engine(DBEngine.database_location)
            Session = sessionmaker(bind=engine)
            session = Session()
            for i in range(14240396,19240396):
                with lock:
                    c = session.query(DBEngine.StoryPage).get(i)
                if c is None:
                    continue
                print(i)
                s=StoriesTextScraper(c)
                s.lock=lock
                s.start()
        except Exception as e:
            pass
