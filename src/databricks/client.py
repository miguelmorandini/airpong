"""Databricks connection and authentication manager for Unity Catalog and Databricks Apps."""

import os
from typing import Optional
from databricks.sdk import WorkspaceClient


class DatabricksClientManager:
    """Manages workspace client connection and credential discovery."""

    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.host = host or os.environ.get("DATABRICKS_HOST")
        self.token = token or os.environ.get("DATABRICKS_TOKEN")
        self._client: Optional[WorkspaceClient] = None
        self._is_connected: Optional[bool] = None

    def is_connected(self) -> bool:
        """Check if connected to active Databricks workspace."""
        if self._is_connected is not None:
            return self._is_connected

        if not self.host or not self.token:
            self._is_connected = False
            return False

        try:
            client = WorkspaceClient(host=self.host, token=self.token)
            # Test ping current user
            user = client.current_user.me()
            self._client = client
            self._is_connected = user is not None
            return bool(self._is_connected)
        except Exception:
            self._is_connected = False
            return False

    @property
    def client(self) -> Optional[WorkspaceClient]:
        if self.is_connected():
            return self._client
        return None
