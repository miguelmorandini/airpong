"""Analytical queries and metric aggregations for ping pong telemetry."""

from typing import Any, Dict, List
import pandas as pd


class PingPongAnalytics:
    """Computes advanced performance analytics on recorded table tennis matches."""

    @staticmethod
    def compute_match_breakdown(shots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute detailed shot metrics for a single match."""
        if not shots:
            return {
                "total_shots": 0,
                "player_shots": 0,
                "ai_shots": 0,
                "player_unforced_errors": 0,
                "player_winners": 0,
                "avg_speed": 0.0,
                "max_speed": 0.0,
            }

        df = pd.DataFrame(shots)

        p_shots = df[df["hitter"] == "PLAYER"]
        ai_shots = df[df["hitter"].isin(["AI", "ROBERVAL"])]

        # Errors: balls hit out or into net
        p_errors = len(p_shots[p_shots["result"].isin(["OUT", "NET_FAULT"])])
        p_winners = len(p_shots[p_shots["result"] == "WINNER"])

        speeds = df["speed_kmh"].dropna()
        avg_speed = float(speeds.mean()) if not speeds.empty else 0.0
        max_speed = float(speeds.max()) if not speeds.empty else 0.0

        return {
            "total_shots": len(df),
            "player_shots": len(p_shots),
            "ai_shots": len(ai_shots),
            "player_unforced_errors": p_errors,
            "player_winners": p_winners,
            "avg_speed": round(avg_speed, 1),
            "max_speed": round(max_speed, 1),
        }
