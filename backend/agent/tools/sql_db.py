import os
import sqlite3
import asyncio
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from core.logger import log

DB_PATH = "mock_database.db"

class SQLInput(BaseModel):
    query: str = Field(description="A valid SQLite SQL query to execute against the 'employees' table.")

def _init_db():
    if not os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary REAL)''')
            c.execute("INSERT INTO employees (name, salary) VALUES ('Alice', 75000)")
            c.execute("INSERT INTO employees (name, salary) VALUES ('Bob', 82000)")
            c.execute("INSERT INTO employees (name, salary) VALUES ('Evan', 60000)")
            conn.commit()
            conn.close()
            log.info("Initialized mock SQL database.")
        except Exception as e:
            log.exception(f"Failed to initialize database: {e}")

_init_db()

def _run_query(query: str) -> str:
    """Synchronous SQL execution to be run in a thread."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(query)
        results = c.fetchall()
        conn.close()
        return str(results) if results else "No results found."
    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}. Please correct your SQL syntax and try again."
    except Exception as e:
        return f"Error executing query: {str(e)}"

@tool("sql_db_tool", args_schema=SQLInput)
async def sql_db_tool(query: str) -> str:
    """Useful for querying the internal company employees SQL database to answer questions about employee salaries. The table is 'employees', columns are 'id', 'name', 'salary'."""
    log.info(f"Executing SQL Query (async thread): {query}")
    # Run the blocking sqlite I/O in a worker thread to keep the event loop free.
    return await asyncio.to_thread(_run_query, query)
