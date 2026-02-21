"""
Text2SQL Tool for Database Querying

This module provides a tool that can convert natural language queries into SQL
and execute them against uploaded database files (SQLite, CSV, etc.).
"""

import os
import sqlite3
import pandas as pd
import json
import logging
import tempfile
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class Text2SQLTool:
    """
    A tool that converts natural language queries to SQL and executes them
    against uploaded database files.
    """

    def __init__(self, config=None):
        """
        Initialize the Text2SQL tool.

        Args:
            config: Configuration object containing LLM settings
        """
        self.name = "text2sql"
        self.description = "Convert natural language queries to SQL and execute against uploaded databases"
        self.config = config
        self.databases = {}  # Store database connections and metadata
        self.temp_dir = Path(tempfile.gettempdir()) / "deep_research_databases"
        self.temp_dir.mkdir(exist_ok=True)

    def _get_llm_client(self):
        """Get the LLM client for text2sql conversion."""
        try:
            from llm_clients import get_llm_client, get_model_response

            if self.config:
                from src.graph import get_configurable

                configurable = get_configurable(self.config)
                provider = configurable.llm_provider
                model = configurable.llm_model
            else:
                provider = os.environ.get("LLM_PROVIDER", "azure")
                model = os.environ.get("LLM_MODEL", None)

            return get_llm_client(provider, model)
        except Exception as e:
            logger.error(f"Error getting LLM client: {e}")
            return None

    def upload_database(
        self, file_content: bytes, filename: str, file_type: str = None
    ) -> str:
        """
        Upload and process a database file.

        Args:
            file_content: The database file content as bytes
            filename: Original filename
            file_type: File type (sqlite, csv, etc.)

        Returns:
            Database ID for future queries
        """
        try:
            # Generate unique database ID
            db_id = str(uuid.uuid4())

            # Determine file type from extension if not provided
            if not file_type:
                ext = Path(filename).suffix.lower()
                if ext == ".db" or ext == ".sqlite" or ext == ".sqlite3":
                    file_type = "sqlite"
                elif ext == ".csv":
                    file_type = "csv"
                elif ext == ".json":
                    file_type = "json"
                else:
                    raise ValueError(f"Unsupported file type: {ext}")

            # Save file to temp directory
            temp_file_path = self.temp_dir / f"{db_id}_{filename}"
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            # Process based on file type
            if file_type == "sqlite":
                metadata = self._process_sqlite_file(temp_file_path, db_id)
            elif file_type == "csv":
                metadata = self._process_csv_file(temp_file_path, db_id)
            elif file_type == "json":
                metadata = self._process_json_file(temp_file_path, db_id)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Store database metadata
            self.databases[db_id] = {
                "id": db_id,
                "filename": filename,
                "file_type": file_type,
                "file_path": str(temp_file_path),
                "metadata": metadata,
                "uploaded_at": datetime.now().isoformat(),
            }

            logger.info(f"Successfully uploaded database {filename} with ID {db_id}")
            return db_id

        except Exception as e:
            logger.error(f"Error uploading database {filename}: {e}")
            raise

    def _process_sqlite_file(self, file_path: Path, db_id: str) -> Dict[str, Any]:
        """Process SQLite file and extract schema information."""
        try:
            conn = sqlite3.connect(str(file_path))
            cursor = conn.cursor()

            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            # Get schema for each table
            schema = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()

                # Get sample data (first 3 rows)
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_data = cursor.fetchall()

                schema[table] = {
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "not_null": bool(col[3]),
                            "default": col[4],
                            "primary_key": bool(col[5]),
                        }
                        for col in columns
                    ],
                    "sample_data": sample_data,
                    "row_count": self._get_table_row_count(cursor, table),
                }

            conn.close()

            return {"type": "sqlite", "tables": tables, "schema": schema}

        except Exception as e:
            logger.error(f"Error processing SQLite file: {e}")
            raise

    def _process_csv_file(self, file_path: Path, db_id: str) -> Dict[str, Any]:
        """Process CSV file and extract schema information."""
        try:
            # Read CSV file
            df = pd.read_csv(str(file_path))

            # Convert to SQLite for easier querying
            sqlite_path = self.temp_dir / f"{db_id}_converted.db"
            conn = sqlite3.connect(str(sqlite_path))
            df.to_sql("data", conn, if_exists="replace", index=False)

            # Get schema information
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(data)")
            columns = cursor.fetchall()

            # Get sample data
            cursor.execute("SELECT * FROM data LIMIT 3")
            sample_data = cursor.fetchall()

            schema = {
                "data": {
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "not_null": bool(col[3]),
                            "default": col[4],
                            "primary_key": bool(col[5]),
                        }
                        for col in columns
                    ],
                    "sample_data": sample_data,
                    "row_count": len(df),
                }
            }

            conn.close()

            return {
                "type": "csv",
                "tables": ["data"],
                "schema": schema,
                "original_csv_path": str(file_path),
                "sqlite_path": str(sqlite_path),
            }

        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise

    def _process_json_file(self, file_path: Path, db_id: str) -> Dict[str, Any]:
        """Process JSON file and extract schema information."""
        try:
            # Read JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Convert to SQLite for easier querying
            sqlite_path = self.temp_dir / f"{db_id}_converted.db"
            conn = sqlite3.connect(str(sqlite_path))

            # Handle different JSON structures
            if isinstance(json_data, list):
                # Array of objects - create a single table
                if json_data:
                    df = pd.DataFrame(json_data)
                    df.to_sql("data", conn, if_exists="replace", index=False)
                    table_name = "data"
                else:
                    # Empty array
                    conn.execute("CREATE TABLE data (id INTEGER PRIMARY KEY)")
                    table_name = "data"
            elif isinstance(json_data, dict):
                # Object - create tables for each key
                tables = []
                for key, value in json_data.items():
                    if isinstance(value, list) and value:
                        # Array of objects
                        df = pd.DataFrame(value)
                        table_name = key.replace(" ", "_").replace("-", "_").lower()
                        df.to_sql(table_name, conn, if_exists="replace", index=False)
                        tables.append(table_name)
                    elif isinstance(value, dict):
                        # Nested object - flatten to single row
                        df = pd.DataFrame([value])
                        table_name = key.replace(" ", "_").replace("-", "_").lower()
                        df.to_sql(table_name, conn, if_exists="replace", index=False)
                        tables.append(table_name)
                    else:
                        # Simple value - create single column table
                        table_name = key.replace(" ", "_").replace("-", "_").lower()
                        conn.execute(f"CREATE TABLE {table_name} (value TEXT)")
                        conn.execute(
                            f"INSERT INTO {table_name} (value) VALUES (?)",
                            (str(value),),
                        )
                        tables.append(table_name)

                if not tables:
                    # Empty object
                    conn.execute("CREATE TABLE data (id INTEGER PRIMARY KEY)")
                    tables = ["data"]
            else:
                # Simple value
                conn.execute("CREATE TABLE data (value TEXT)")
                conn.execute("INSERT INTO data (value) VALUES (?)", (str(json_data),))
                tables = ["data"]

            # Get schema information for all tables
            schema = {}
            cursor = conn.cursor()

            for table_name in tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                # Get sample data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                schema[table_name] = {
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "not_null": bool(col[3]),
                            "default": col[4],
                            "primary_key": bool(col[5]),
                        }
                        for col in columns
                    ],
                    "sample_data": sample_data,
                    "row_count": row_count,
                }

            conn.close()

            return {
                "type": "json",
                "tables": tables,
                "schema": schema,
                "original_json_path": str(file_path),
                "sqlite_path": str(sqlite_path),
            }

        except Exception as e:
            logger.error(f"Error processing JSON file: {e}")
            raise

    def _get_table_row_count(self, cursor, table_name: str) -> int:
        """Get row count for a table."""
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        except:
            return 0

    def list_databases(self) -> List[Dict[str, Any]]:
        """List all uploaded databases."""
        return [
            {
                "id": db["id"],
                "filename": db["filename"],
                "file_type": db["file_type"],
                "uploaded_at": db["uploaded_at"],
                "tables": db["metadata"]["tables"],
                "table_count": len(db["metadata"]["tables"]),
            }
            for db in self.databases.values()
        ]

    def get_database_schema(self, db_id: str) -> Dict[str, Any]:
        """Get schema information for a specific database."""
        if db_id not in self.databases:
            raise ValueError(f"Database {db_id} not found")

        return self.databases[db_id]["metadata"]

    def query_database(self, db_id: str, natural_language_query: str) -> Dict[str, Any]:
        """
        Convert natural language query to SQL and execute it.

        Args:
            db_id: Database ID
            natural_language_query: Natural language query

        Returns:
            Query results and metadata
        """
        try:
            if db_id not in self.databases:
                raise ValueError(f"Database {db_id} not found")

            db_info = self.databases[db_id]
            schema = db_info["metadata"]

            # Generate SQL from natural language
            sql_query = self._generate_sql(natural_language_query, schema)

            # Execute SQL query
            results = self._execute_sql(db_id, sql_query)

            return {
                "query": natural_language_query,
                "sql": sql_query,
                "results": results,
                "database": db_info["filename"],
                "executed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error querying database {db_id}: {e}")
            return {
                "query": natural_language_query,
                "error": str(e),
                "database": self.databases.get(db_id, {}).get("filename", "Unknown"),
                "executed_at": datetime.now().isoformat(),
            }

    def _generate_sql(self, natural_language_query: str, schema: Dict[str, Any]) -> str:
        """Generate SQL query from natural language using LLM."""
        try:
            llm_client = self._get_llm_client()
            if not llm_client:
                raise ValueError("LLM client not available")

            # Create schema description for the LLM
            schema_description = self._format_schema_for_llm(schema)

            prompt = f"""
You are a SQL expert specializing in analytical queries. Convert the following natural language query to SQL.

Database Schema:
{schema_description}

Natural Language Query: {natural_language_query}

CRITICAL INSTRUCTIONS:
1. Generate ANALYTICAL queries with insights, NOT simple SELECT * queries
2. Use aggregations (COUNT, AVG, SUM, MAX, MIN) when appropriate
3. Use GROUP BY for categorical analysis
4. ALWAYS add LIMIT 20 to prevent returning too many rows
5. ORDER BY the most relevant column (usually DESC for aggregates)
6. Return ONLY the SQL query, no explanations or markdown
7. Focus on answering the ANALYTICAL question, not dumping raw data

SQL Query:
"""

            # Handle different LLM client types
            if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
                # OpenAI-style client
                response = llm_client.chat.completions.create(
                    model=getattr(llm_client, "model", "gpt-4"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1,
                )
                sql_query = response.choices[0].message.content.strip()
            elif hasattr(llm_client, "invoke"):
                # LangChain-style client
                from langchain_core.messages import HumanMessage

                response = llm_client.invoke([HumanMessage(content=prompt)])
                sql_query = response.content.strip()
            elif hasattr(llm_client, "predict"):
                # LangChain predict method
                response = llm_client.predict(prompt)
                sql_query = response.strip()
            elif hasattr(llm_client, "__call__"):
                # Callable client
                response = llm_client(prompt)
                sql_query = response.strip()
            elif "ChatVertexAI" in str(type(llm_client)):
                # Google Vertex AI client
                from langchain_core.messages import HumanMessage

                response = llm_client.invoke([HumanMessage(content=prompt)])
                sql_query = response.content.strip()
            else:
                # Fallback: try to use the client as a string
                logger.warning(f"Unknown LLM client type: {type(llm_client)}")
                # For now, return a simple SQL query as fallback
                sql_query = "SELECT * FROM customers WHERE state = 'CA'"

            # Clean up the SQL query (remove markdown formatting if present)
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]

            sql_query = sql_query.strip()

            logger.info(f"Generated SQL: {sql_query}")
            return sql_query

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise

    def _format_schema_for_llm(self, schema: Dict[str, Any]) -> str:
        """Format database schema for LLM consumption."""
        schema_text = f"Database Type: {schema['type']}\n\n"

        for table_name, table_info in schema["schema"].items():
            schema_text += f"Table: {table_name}\n"
            schema_text += f"  Rows: {table_info['row_count']}\n"
            schema_text += "  Columns:\n"

            for col in table_info["columns"]:
                schema_text += f"    - {col['name']} ({col['type']})"
                if col["primary_key"]:
                    schema_text += " [PRIMARY KEY]"
                if col["not_null"]:
                    schema_text += " [NOT NULL]"
                schema_text += "\n"

            # Add sample data
            if table_info["sample_data"]:
                schema_text += "  Sample Data:\n"
                for i, row in enumerate(
                    table_info["sample_data"][:2]
                ):  # Show first 2 rows
                    schema_text += f"    Row {i+1}: {row}\n"

            schema_text += "\n"

        return schema_text

    def _execute_sql(self, db_id: str, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query against the database."""
        try:
            db_info = self.databases[db_id]

            # Determine which database file to use
            if db_info["file_type"] == "sqlite":
                db_path = db_info["file_path"]
            elif db_info["file_type"] in ["csv", "json"]:
                db_path = db_info["metadata"]["sqlite_path"]
            else:
                raise ValueError(f"Unsupported database type: {db_info['file_type']}")

            # Execute query
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Execute the query
            cursor.execute(sql_query)

            # Get results
            if sql_query.strip().upper().startswith("SELECT"):
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]

                # Convert to list of dictionaries for easier handling
                formatted_results = []
                for row in results:
                    formatted_results.append(dict(zip(columns, row)))

                return {
                    "type": "select",
                    "columns": columns,
                    "rows": formatted_results,
                    "row_count": len(formatted_results),
                }
            else:
                # For INSERT, UPDATE, DELETE queries
                conn.commit()
                return {"type": "modify", "rows_affected": cursor.rowcount}

        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            raise
        finally:
            if "conn" in locals():
                conn.close()

    def delete_database(self, db_id: str) -> bool:
        """Delete a database and its files."""
        try:
            if db_id not in self.databases:
                return False

            db_info = self.databases[db_id]

            # Delete files
            files_to_delete = [db_info["file_path"]]
            if db_info["file_type"] == "csv" and "sqlite_path" in db_info["metadata"]:
                files_to_delete.append(db_info["metadata"]["sqlite_path"])

            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except:
                    pass  # File might already be deleted

            # Remove from databases dict
            del self.databases[db_id]

            logger.info(f"Deleted database {db_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting database {db_id}: {e}")
            return False

    def _run(self, query: str, db_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Main execution method for the tool.

        Args:
            query: Natural language query
            db_id: Database ID (optional, will use first available if not provided)

        Returns:
            Query results
        """
        try:
            # If no db_id provided, use the first available database
            if not db_id and self.databases:
                db_id = list(self.databases.keys())[0]
            elif not db_id:
                return {
                    "error": "No databases available. Please upload a database file first.",
                    "query": query,
                }

            return self.query_database(db_id, query)

        except Exception as e:
            logger.error(f"Error in Text2SQL tool execution: {e}")
            return {"error": str(e), "query": query}
