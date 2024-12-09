import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, Float, String

load_dotenv()

url = os.getenv("DATABASE_URL")
engine = create_engine(url, pool_pre_ping=True)

Base = declarative_base()

class Data(Base):
    __tablename__ = 'Data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ed = Column(Integer)
    tenure = Column(Integer)
    employ = Column(Integer)
    income = Column(Float)
    marital = Column(Integer)
    address = Column(Integer)
    reside = Column(Integer)
    prediction = Column(Integer)
    

def add_data(data, prediction):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        new_data = Data(
            ed = data['ed'][0],
            tenure = data['tenure'][0],
            employ = data['employ'][0],
            reside = data['reside'][0],
            income = data['income'][0],
            marital = data['marital'][0],
            address = data['address'][0],
            prediction = prediction
        )
        session.add(new_data)
        session.commit()
        print("ok")
        return 1
    except Exception as e:
        session.rollback()
        print(e)
    finally:
        session.close()