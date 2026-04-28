import os
import pandas as pd

def load_parquet(local_path, blob_path=None, container="riskml-data"):
    """Load parquet from Azure Blob if configured, else from local path."""
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn and blob_path:
        try:
            from azure.storage.blob import BlobClient
            blob = BlobClient.from_connection_string(
                conn_str=conn, container_name=container, blob_name=blob_path
            )
            return pd.read_parquet(blob.download_blob())
        except Exception:
            pass
    return pd.read_parquet(local_path)