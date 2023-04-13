#! /usr/bin/python3

"""
Author: Genevieve LaLonde

A simple database connector.
Created as part of an interview coding challenge.
"""

import mysql.connector 
import os


def getDatabaseConnection():
    # Get a connection to the database.

    conn = mysql.connector.connect(
      host = os.environ.get('HOST'),
      user = os.environ.get('USER'),
      password = os.environ.get('PASSWORD'),
      port = os.environ.get('PORT'),
      database = os.environ.get('DATABASE')
      )
    return conn

def printRows(result):
    # Basic print of data as raw tuples.
    for row in result:
        print(row)


def convertRows(result, colNames):
    # Convert the raw result from database blob to a dictionary
    data = []
    for _, row in enumerate(result):
        entry = {}
        for j in range(len(colNames)):
            entry[colNames[j]] = row[j]
        data.append(entry)
    return data


def queryDatabase(conn, query='', convert=True):
    try:
        if conn.is_connected():
            cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()

        # Automatically pull the column names from result for conversion
        colNames = [i[0] for i in cursor.description]

        if result:
            result = convertRows(result, colNames)
        else:
            print('Not connected.')
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        conn.close()
    return result


def getDataDump(n=1073741824):
    # Pull up to n rows from the database.
    # Reorders id to first, so it can be used as object id when converting to local object.

    # TODO enable batch, offset for pagination/batching

    conn = getDatabaseConnection()

    # Overwrite col names from db to match up with the .csv file
    query = f"""
        select 
            team_id,
            id as entry_id, 
            title, 
            entry, 
            data_source,
            sentiment as annotated_sentiment
        from feedback_entries
        limit {n};
        """
    return queryDatabase(conn, query)




