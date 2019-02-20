import DBEngine
import multiprocessing
from StoriesScraper import StoriesScraper

if __name__ == '__main__':
    DBEngine.initiate_database()
    cpu_count = multiprocessing.cpu_count()
    processes = [StoriesScraper(i,cpu_count) for i in range(cpu_count)]
    for p in processes:
        p.start()