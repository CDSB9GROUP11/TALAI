
import sys
sys.path.append('/content/drive/MyDrive/CDS Collab Notebooks/Capstone/CodePipelines3/')

import pandas as pd
import os
import logging
import ast
import json
from time import sleep

from dotenv import load_dotenv
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra import OperationTimedOut

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AstraConnector")

def setup_astra_session(
    bundle_path='/content/drive/MyDrive/CDS Collab Notebooks/Capstone/CodePipelines3/secure-connect-resumes-database.zip',
    keyspace='embeddings',
    max_retries=3,
    retry_delay=5
):
    # Load credentials
    client_id = os.getenv('ASTRA_RESUME_CLIENT')
    client_secret = os.getenv('ASTRA_RESUME_SECRET')

    if not client_id or not client_secret:
        raise ValueError("❌ Missing ASTRA credentials. Set ASTRA_RESUME_CLIENT and ASTRA_RESUME_SECRET.")

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"❌ Secure connect bundle not found at: {bundle_path}")

    cloud_config = {'secure_connect_bundle': bundle_path}
    auth_provider = PlainTextAuthProvider(client_id, client_secret)

    # Retry logic
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🔌 Connecting to Astra DB (Attempt {attempt})...")
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect()

            # Optional health check
            row = session.execute("SELECT release_version FROM system.local").one()
            if row:
                logger.info(f"✅ Connected to Astra DB. Release version: {row[0]}")
            else:
                logger.warning("⚠️ Connected but failed to fetch release version.")

            # Set keyspace
            session.set_keyspace(keyspace)
            logger.info(f"📦 Keyspace set to: {keyspace}")
            return session

        except OperationTimedOut as e:
            logger.warning(f"⏱️ Timeout during connection: {e}")
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")

        sleep(retry_delay)

    raise ConnectionError("🚫 Failed to connect to Astra DB after multiple attempts.")

def create_resume_table():
    query = """CREATE TABLE IF NOT EXISTS embeddings.section_embed (
    resume_id TEXT,
    embedding_vectors LIST<FROZEN<TUPLE<FLOAT>>>,
    metadata MAP<TEXT, TEXT>,
    section_text TEXT,
    PRIMARY KEY (resume_id));"""
    try:
        session.execute(query)
    except Exception as e:
        print(e)

session = setup_astra_session()
create_resume_table()

def fetch_existing_resumes(file_ids):
    logger.info(f"Enter fetching existing resumes: {file_ids}")
    existing_data = []
    found_ids = set()

    select_query = "SELECT resume_id, embedding_vectors, metadata, section_text FROM embeddings.section_embed WHERE resume_id = %s"

    for resume_id in file_ids:
        try:
            result = session.execute(select_query, [resume_id])
            row = result.one()
            if row:
                logger.info("Found data : %s"%resume_id)
                existing_data.append({
                    "resume_id": row.resume_id,
                    "embedding_vectors": row.embedding_vectors,
                    "metadata": row.metadata,
                    "section_text": row.section_text
                })
                found_ids.add(resume_id)
        except Exception as e:
            print(f"Error querying resume_id '{resume_id}': {e}")

    # Create DataFrame from existing rows
    existing_df = pd.DataFrame(existing_data) if existing_data else None

    # Identify missing IDs
    missing_ids = [rid for rid in file_ids if rid not in found_ids]

    return existing_df, missing_ids

def safe_literal_eval(val):
    try:
        if not val or str(val).lower() in ['nan', 'none', 'null']:
            return []
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as e:
        print(f"Skipping malformed vector: {val} → {e}")
        return []

def insert_resume_data(df):
    select_query = "SELECT resume_id FROM embeddings.section_embed WHERE resume_id = %s"
    insert_query = """INSERT INTO embeddings.section_embed (
        resume_id, embedding_vectors, metadata, section_text
    ) VALUES (%s, %s, %s, %s)"""

    for _, row in df.iterrows():
        resume_id = row['id']
        try:
            # Check existence
            result = session.execute(select_query, [resume_id])
            if result.one():
                print(f"Skipped: resume_id '{resume_id}' already exists.")
                continue

            # Normalize metadata
            metadata_clean = {k: str(v) for k, v in row['metadata'].items()}

            # Ensure vector is serializable
            embedding_vectors = row['vector'] if isinstance(row['vector'], list) else safe_literal_eval(row['vector'])

            resume_id = row['id']

            section_text = row['section_text'] # Long string

            # Insert new row
            session.execute(insert_query, [resume_id, embedding_vectors, metadata_clean, section_text])
            print(f"Inserted: resume_id '{resume_id}'")

        except Exception as e:
            print(f"Error processing resume_id '{resume_id}': {e}")
