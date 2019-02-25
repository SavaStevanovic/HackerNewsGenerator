from sqlalchemy import create_engine, MetaData, Table, ForeignKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.sql import table
from sqlalchemy.dialects.sqlite import TEXT
from sqlalchemy import create_engine, MetaData, Table

database_location = "sqlite:///sqlite_data.db"
StoryPageTable = "HackerNewsStoryUrls"
StoryTextsTable = "HackerNewsStoryTexts"
StoryPageTableID = StoryPageTable+'.id'
# module init
Base = declarative_base()
engine = create_engine(database_location)
metadata = MetaData(bind=engine)


def initiate_database():
    Base.metadata.create_all(engine)


class StoryPage(Base):
    '''
    First argument is id of database row.
    Second argument is link which returns 
    html represented by third argument.
    '''
    __tablename__ = StoryPageTable

    id = Column(Integer, primary_key=True)
    title = Column(TEXT)
    url = Column(TEXT)


class StoryText(Base):
    '''
    First argument is id of database row.
    Second argument is link which returns 
    html represented by third argument.
    '''
    __tablename__ = StoryTextsTable

    id = Column(Integer, primary_key=True)
    id_source = Column(Integer, ForeignKey(StoryPageTableID))
    text = Column(TEXT)
