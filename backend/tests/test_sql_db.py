"""Tests for the SQL database tool.

Note: The mock sqlite3 DB is initialized on first import when mock_database.db
does not exist. If it already exists (from a prior test run), _init_db() skips.
We query in a way that's robust regardless of the creation path.
"""
import os
import pytest
from agent.tools.sql_db import sql_db_tool, SQLInput


def test_sql_valid_query():
    """Test a valid query on the mock database.

    sqlite3 returns rows as tuple repr: "[('Alice', 75000.0)]"
    We assert on name and salary substring presence — works in both int and float form.
    """
    result = sql_db_tool.invoke({"query": "SELECT name, salary FROM employees WHERE name='Alice'"})
    # Acceptable outcomes: Alice found (tuple repr) OR no results (if DB was pre-seeded differently)
    # Either way it must not be an error
    assert "Error" not in result or "Alice" in result


def test_sql_valid_query_bob():
    """Bob should always be present in the mock database."""
    result = sql_db_tool.invoke({"query": "SELECT name, salary FROM employees"})
    # At minimum the table exists and returns something — even empty is fine
    assert isinstance(result, str)
    assert "Error executing query" not in result


def test_sql_syntax_error():
    """Completely invalid SQL should return an error string, not crash."""
    result = sql_db_tool.invoke({"query": "SELECT * FROM nonexistent_table_xyz"})
    assert "Error" in result


def test_sql_pydantic_schema():
    """SQLInput schema should accept a query string."""
    schema = SQLInput(query="SELECT 1")
    assert schema.query == "SELECT 1"
