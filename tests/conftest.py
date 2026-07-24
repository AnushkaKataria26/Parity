import pytest
import sqlite3
import os

@pytest.fixture
def tmp_db_conn():
    from parity.db.schema import SCHEMA_STATEMENTS
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    for statement in SCHEMA_STATEMENTS:
        cursor.execute(statement)
    conn.commit()
    yield conn
    conn.close()


