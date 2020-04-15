import numpy as np
import pandas as pd
import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None


def select_all(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM Jobs")

    rows = cur.fetchall()
    arr = []
    for row in rows:
        arr.append(row)
    print(len(arr))
    arr = np.array(arr)
    print(arr.shape)
    df = pd.DataFrame(np.array(arr))
    df.to_csv('/Users/thangnguyen/hust_project/cloud_autoscaling/data/input_data/grid_data/anon_jobs.csv',
              index=False, header=None)


def read_sqllite(db_file):
    conn = create_connection(db_file)
    with conn:
        print("1. Query all table:")
        select_all(conn)
