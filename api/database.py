from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy as sqlal
import pandas as pd

SQLALCHEMY_DATABASE_URL = "mysql+mysqlconnector://root:1234@localhost:3306/goodreads"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

connection = engine.connect()
metadata = sqlal.MetaData()
db_books = sqlal.Table('books', metadata, autoload=True, autoload_with=engine)
db_ratings = sqlal.Table('ratings', metadata, autoload=True, autoload_with=engine)
db_books_infos = sqlal.Table('books_infos', metadata, autoload=True, autoload_with=engine)

