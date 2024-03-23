"""
A module that provides an interface to interact with stock bot database
"""

import sqlite3
from datetime import datetime

current_date = datetime.now()
formatted_date = current_date.strftime('%m/%d/%y')

def singleton(cls):
  instances = {}
  def get_instance():
    if cls not in instances:
      instances[cls] = cls()
    return instances[cls]
  return get_instance

class DatabaseDriver(object):
  def __init__(self):
    self.conn = sqlite3.connect('logs.db', check_same_thread=False)
    self.conn.execute('PRAGMA foreign_keys = ON;')
    self.create_logs_table()
  
  def create_logs_table(self):
    """
    Create table that contains performance logs
    """
    self.conn.execute("""CREATE TABLE IF NOT EXISTS logs(
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date TEXT NOT NULL,
                      equity FLOAT NOT NULL);
                      """)
  
  def delete_logs_table(self):
    """
    Delete table that contains performance logs
    """
    self.conn.execute('DROP TABLE IF EXISTS logs;')
  
  def insert_log(self, equity):
    """
    Insert a performance log into the logs table; if the logs table already contains an entry with today's date, simply update the existing entry with the new equity instead of inserting a new entry
    """
    existing_log_id = self.get_log_id_by_date(formatted_date)
    if existing_log_id is not None:
      self.conn.execute('UPDATE logs SET equity = ? WHERE id = ?;', (equity, existing_log_id))
      self.conn.commit()
      return existing_log_id
    else:
      cursor = self.conn.execute('INSERT INTO logs (date, equity) VALUES (?, ?);', (formatted_date, equity))
      self.conn.commit()
      return cursor.lastrowid
    
  def get_log_count(self):
    """
    Retrieve the number of stored logs
    """
    cursor = self.conn.execute('SELECT COUNT(*) FROM logs;')
    row = cursor.fetchone()
    return row[0] if row is not None and row[0] is not None else None

  def get_log_by_id(self, log_id):
    """
    Retrieve the performance log with the given log ID
    """
    cursor = self.conn.execute('SELECT * FROM logs WHERE id = ?;', (log_id,))
    for row in cursor:
      return {'id': row[0], 'date': row[1], 'equity': row[2]}
    return None
  
  def get_log_id_by_date(self, log_date):
    """
    Retrieve the log ID of the performance log created on the given date
    """
    cursor = self.conn.execute('SELECT id FROM logs WHERE date = ?;', (log_date,))
    row = cursor.fetchone()
    return row[0] if row is not None and row[0] is not None else None
  
  def get_current_id(self):
    """
    Retrieve the log ID of today's entry in the logs table
    """
    cursor = self.conn.execute('SELECT MAX(id) FROM logs;')
    row = cursor.fetchone()
    if row is not None and row[0] is not None:
      # If the logs table already contains an entry from today, simply return that entry's ID; otherwise return one more than the ID of the previous day's entry
      if self.get_log_id_by_date(formatted_date) is not None:
        return row[0]
      else:
        return row[0] + 1
    else:
      # Return 1 for the log ID of today's entry if there are currently no entries in the logs table
      return 1

DatabaseDriver = singleton(DatabaseDriver)