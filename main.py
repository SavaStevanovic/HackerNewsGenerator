import DBEngine
import multiprocessing
from StoriesScraper import StoriesScraper
from StoriesTextScraper import StoriesTextScraper
scrapeUrls = False
scrapeText = True

if __name__ == '__main__':
    DBEngine.initiate_database()
    cpu_count = multiprocessing.cpu_count()
    if scrapeUrls:
        processes = [StoriesScraper(i, cpu_count) for i in range(cpu_count)]
        for p in processes:
            p.start()
    if scrapeText:
        processes = [StoriesTextScraper(i, cpu_count) for i in range(cpu_count)]
        for p in processes:
            p.start()