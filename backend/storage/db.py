"""Database connection helpers for persistence backends."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from typing import Generator


def postgres_driver_available() -> bool:
    """Return True when psycopg2 is installed and importable."""
    try:
        import_module("psycopg2")
        return True
    except Exception:
        return False


@contextmanager
def postgres_connection(dsn: str) -> Generator[object, None, None]:
    """Yield a Postgres connection using psycopg2 when available."""
    try:
        psycopg2 = import_module("psycopg2")
    except Exception as exc:
        raise RuntimeError("psycopg2 is required for postgres persistence") from exc

    connection = psycopg2.connect(dsn)
    try:
        yield connection
    finally:
        connection.close()


def ensure_incident_schema(dsn: str) -> None:
    """Create the incidents table if it does not already exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS incident_records (
        incident_id TEXT PRIMARY KEY,
        payload JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """

    with postgres_connection(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(create_table_sql)
        connection.commit()
