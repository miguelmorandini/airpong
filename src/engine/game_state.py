"""Match state machine, ITTF scoring rules, and telemetry collector."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.config import RulesConfig
from src.engine.physics import Ball, PhysicsEvent, PhysicsEngine
from src.engine.paddle import Paddle
from src.engine.ai import TableTennisAI


class MatchStatus(str, Enum):
    READY = "READY"
    SERVING = "SERVING"
    RALLY = "RALLY"
    POINT_SCORED = "POINT_SCORED"
    GAME_OVER = "GAME_OVER"


@dataclass
class ShotRecord:
    """Telemetry data for an individual table tennis shot."""
    shot_id: str
    match_id: str
    shot_number: int
    hitter: str            # "PLAYER" or "AI"
    shot_type: str         # "SERVE", "FOREHAND", "BACKHAND", "SMASH", "DRIVE"
    speed_kmh: float
    spin: str
    landing_x: Optional[float] = None
    landing_z: Optional[float] = None
    result: str = "IN_PLAY"  # "IN_PLAY", "WINNER", "OUT", "NET_FAULT"
    timestamp: float = field(default_factory=time.time)


@dataclass
class MatchStats:
    """Aggregated match statistics for Databricks Delta Lake."""
    match_id: str
    start_time: float
    end_time: Optional[float] = None
    player_score: int = 0
    ai_score: int = 0
    winner: Optional[str] = None
    difficulty: str = "MEDIUM"
    total_rallies: int = 0
    longest_rally: int = 0
    player_shots_count: int = 0
    ai_shots_count: int = 0
    avg_speed_kmh: float = 0.0
    player_smashes: int = 0
    shots: List[ShotRecord] = field(default_factory=list)


class GameStateMachine:
    """Manages the full lifecycle of an ITTF ping pong match."""

    def __init__(
        self,
        difficulty: str = "MEDIUM",
        rules: Optional[RulesConfig] = None,
    ):
        self.rules = rules or RulesConfig()
        self.difficulty = difficulty
        self.status = MatchStatus.READY
        self.server: str = "PLAYER"
        self.player_score: int = 0
        self.ai_score: int = 0
        self.current_rally_count: int = 0
        self.announcement: str = "FECHE A MAO OU ESPACO PARA SACAR!"
        self.announcement_timer: float = 4.0

        self.match_id = str(uuid.uuid4())
        self.stats = MatchStats(
            match_id=self.match_id,
            start_time=time.time(),
            difficulty=difficulty,
        )

        self.physics = PhysicsEngine()
        self.ball = Ball()
        self.player_paddle = Paddle(is_player=True)
        self.ai_paddle = Paddle(is_player=False)
        self.ai = TableTennisAI(difficulty=difficulty)

        self.current_shot: Optional[ShotRecord] = None
        self._shots_speeds: List[float] = []

    def start_new_match(self, difficulty: Optional[str] = None):
        """Reset match to 0-0 and generate new telemetry session."""
        if difficulty:
            self.difficulty = difficulty
            self.ai.set_difficulty(difficulty)

        self.match_id = str(uuid.uuid4())
        self.stats = MatchStats(
            match_id=self.match_id,
            start_time=time.time(),
            difficulty=self.difficulty,
        )
        self.player_score = 0
        self.ai_score = 0
        self.server = "PLAYER"
        self.current_rally_count = 0
        self.status = MatchStatus.READY
        self.announcement = "NEW MATCH! PLAYER SERVES"
        self.announcement_timer = 2.5
        self._shots_speeds.clear()
        self.prepare_serve()

    def prepare_serve(self):
        """Position ball for serve."""
        self.physics.reset_serve(self.ball, server=self.server)
        self.status = MatchStatus.SERVING
        self.current_rally_count = 0

    def trigger_serve(self):
        """Launch serve."""
        if self.status in (MatchStatus.READY, MatchStatus.SERVING):
            self.prepare_serve()
            self.status = MatchStatus.RALLY
            self.announcement = "PLAY!"
            self.announcement_timer = 1.0

            # Log serve shot
            shot = ShotRecord(
                shot_id=str(uuid.uuid4()),
                match_id=self.match_id,
                shot_number=1,
                hitter=self.server,
                shot_type="SERVE",
                speed_kmh=self.ball.speed_kmh,
                spin="NONE",
            )
            self.current_shot = shot
            self.stats.shots.append(shot)
            self._shots_speeds.append(self.ball.speed_kmh)

    def update(self, dt: float = 1.0 / 60.0):
        """Update match frame: physics, AI, rules and scoring."""
        if self.announcement_timer > 0:
            self.announcement_timer -= dt

        if self.status == MatchStatus.GAME_OVER:
            return

        if self.status == MatchStatus.SERVING:
            # Auto serve for AI
            if self.server == "AI" and self.announcement_timer <= 0:
                self.trigger_serve()
            return

        if self.status == MatchStatus.RALLY:
            # Update AI paddle behavior
            self.ai.update(self.ai_paddle, self.ball, dt)

            # Step physics
            events = self.physics.step(self.ball, self.player_paddle, self.ai_paddle, dt)

            # Process physics events
            for event in events:
                self._handle_event(event)

            # Check fault or out-of-play conditions
            if not self.ball.in_play and self.status == MatchStatus.RALLY:
                self._evaluate_point_outcome()

    def _handle_event(self, event: PhysicsEvent):
        """Process hits, bounces, and net encounters."""
        if event.event_type == "HIT":
            self.current_rally_count += 1
            if self.current_rally_count > self.stats.longest_rally:
                self.stats.longest_rally = self.current_rally_count

            if event.hitter == "PLAYER":
                self.stats.player_shots_count += 1
                if event.details == "SMASH":
                    self.stats.player_smashes += 1
                    self.announcement = "SMASH!"
                    self.announcement_timer = 1.0
            else:
                self.stats.ai_shots_count += 1

            self._shots_speeds.append(event.speed_kmh)

            shot = ShotRecord(
                shot_id=str(uuid.uuid4()),
                match_id=self.match_id,
                shot_number=self.current_rally_count,
                hitter=event.hitter or "PLAYER",
                shot_type=event.details or "DRIVE",
                speed_kmh=event.speed_kmh,
                spin="NONE",
            )
            self.current_shot = shot
            self.stats.shots.append(shot)

        elif event.event_type == "BOUNCE":
            # Record landing position on current shot
            if self.current_shot:
                self.current_shot.landing_x = event.position[0]
                self.current_shot.landing_z = event.position[2]

            # In a serve, ball MUST bounce on server side then receiver side
            if self.ball.is_serve:
                if self.ball.bounce_count_current_hit == 1:
                    # First bounce must be on server's side
                    server_side = "PLAYER" if self.server == "PLAYER" else "AI"
                    if event.details != server_side:
                        self.ball.in_play = False
                        self._award_point("AI" if self.server == "PLAYER" else "PLAYER", "FAULT: SERVE DID NOT BOUNCE ON OWN SIDE")
                elif self.ball.bounce_count_current_hit == 2:
                    # Second bounce must be on opponent's side
                    opponent_side = "AI" if self.server == "PLAYER" else "PLAYER"
                    if event.details != opponent_side:
                        self.ball.in_play = False
                        self._award_point("AI" if self.server == "PLAYER" else "PLAYER", "FAULT: SERVE MISSED OPPONENT TABLE")
                    else:
                        # Successful serve transition into open rally
                        self.ball.is_serve = False
            else:
                # Regular rally: if ball bounces twice on one side without being hit, point goes to opponent
                if self.ball.bounce_count_current_hit >= 2:
                    self.ball.in_play = False
                    scoring_party = "PLAYER" if event.details == "AI" else "AI"
                    self._award_point(scoring_party, "DOUBLE BOUNCE!")

        elif event.event_type == "NET_FAULT":
            self.ball.in_play = False
            hitter = event.hitter or self.server
            if hitter == "PLAYER":
                self._award_point("AI", "BOLA NA REDE! PONTO DA IA")
            else:
                self._award_point("PLAYER", "IA ACERTOU A REDE! PONTO SEU!")

        elif event.event_type in ("NET", "NET_CLIP"):
            self.announcement = "RASPOU NA REDE!"
            self.announcement_timer = 1.0

    def _evaluate_point_outcome(self):
        """When ball goes out of play, award point according to ITTF rules."""
        hitter = self.ball.last_hitter or self.server

        # If ball bounced on opponent's side and then went out -> Winner for the hitter!
        if hitter == "PLAYER":
            if self.ball.bounced_ai_side:
                self._award_point("PLAYER", "WINNER POINT!")
            else:
                self._award_point("AI", "OUT! POINT FOR AI")
        else:
            if self.ball.bounced_player_side:
                self._award_point("AI", "WINNER POINT FOR AI!")
            else:
                self._award_point("PLAYER", "AI OUT! POINT FOR PLAYER")

    def _award_point(self, winner: str, reason: str):
        """Add point, check game over, update serve rotation."""
        self.status = MatchStatus.POINT_SCORED
        self.stats.total_rallies += 1

        if winner == "PLAYER":
            self.player_score += 1
            self.announcement = f"NICE POINT! {reason}"
        else:
            self.ai_score += 1
            self.announcement = f"POINT AI! {reason}"

        self.announcement_timer = 2.2

        if self.current_shot:
            self.current_shot.result = "WINNER" if ("WINNER" in reason) else "OUT"

        # Update average shot speed
        if self._shots_speeds:
            self.stats.avg_speed_kmh = sum(self._shots_speeds) / len(self._shots_speeds)

        # Check win condition
        p_score = self.player_score
        a_score = self.ai_score
        to_win = self.rules.POINTS_TO_WIN
        diff = self.rules.MIN_POINT_DIFFERENCE

        if (p_score >= to_win or a_score >= to_win) and abs(p_score - a_score) >= diff:
            self.status = MatchStatus.GAME_OVER
            self.stats.end_time = time.time()
            self.stats.player_score = p_score
            self.stats.ai_score = a_score
            match_winner = "PLAYER" if p_score > a_score else "AI"
            self.stats.winner = match_winner
            self.announcement = f"GAME OVER! {match_winner} WINS ({p_score} - {a_score})!"
            self.announcement_timer = 5.0
            return

        # Check deuce
        if p_score >= 10 and a_score >= 10:
            if p_score == a_score:
                self.announcement += " [DEUCE]"
            elif p_score > a_score:
                self.announcement += " [ADVANTAGE PLAYER]"
            else:
                self.announcement += " [ADVANTAGE AI]"

        # Update serve rotation (every 2 points, or every 1 point in deuce)
        total_points = p_score + a_score
        if p_score >= 10 and a_score >= 10:
            # In deuce, serve alternates every single point
            self.server = "AI" if total_points % 2 == 1 else "PLAYER"
        else:
            # Alternates every 2 points
            rotation_idx = (total_points // self.rules.SERVES_PER_ROTATION) % 2
            self.server = "PLAYER" if rotation_idx == 0 else "AI"

        # Reset for next serve
        self.prepare_serve()
