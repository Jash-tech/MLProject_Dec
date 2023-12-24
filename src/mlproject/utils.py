import os
import sys
import pandas as pd
import pymysql

import numpy as np


from src.mlproject.logger import logging
from src.mlproject.exception import CustomException

from dotenv import load_dotenv
load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Connecting and Reading Data Begins")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db

        )
        logging.info("Connection Established")
        df=pd.read_sql_query("select * from college.student_file",mydb)
        df.head(5)

        return df

    except Exception as e:
        raise CustomException(e,sys)
    





