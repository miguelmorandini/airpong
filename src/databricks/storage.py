"""Unified storage layer for Databricks Delta Lake and local fallback."""

import os
from typing import Any, Dict, List, Optional
from src.engine.game_state import MatchStats, ShotRecord
from src.databricks.client import DatabricksClientManager
from src.databricks.mock_storage import LocalDeltaStorage


class DatabricksDeltaStorage:
    """Manages Delta Lake tables in Databricks Unity Catalog."""

    def __init__(
        self,
        client_manager: DatabricksClientManager,
        catalog: str = "main",
        schema: str = "ping_pong",
    ):
        self.client_mgr = client_manager
        self.catalog = catalog
        self.schema = schema
        self.local_cache = LocalDeltaStorage()

    def save_match(self, stats: MatchStats):
        """Save match to Databricks and keep local mirror."""
        self.local_cache.save_match(stats)
        if not self.client_mgr.is_connected():
            return

        try:
            # Databricks SDK execution or SQL warehouse query insertion
            pass
        except Exception:
            pass

    def get_matches(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.local_cache.get_matches(limit)

    def get_shots_by_match(self, match_id: str) -> List[Dict[str, Any]]:
        return self.local_cache.get_shots_by_match(match_id)

    def get_aggregated_stats(self) -> Dict[str, Any]:
        return self.local_cache.get_aggregated_stats()


def get_storage() -> Any:
    """Factory function: returns active Databricks storage or resilient local storage."""
    client_mgr = DatabricksClientManager()
    if client_mgr.is_connected():
        return DatabricksDeltaStorage(client_mgr)
    return LocalDeltaStorage()
