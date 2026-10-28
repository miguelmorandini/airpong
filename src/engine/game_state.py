"""Match state machine, ITTF scoring rules, and telemetry collector."""

import math
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
    hitter: str            # "PLAYER" or "ROBERVAL" / "AI"
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

    @property
    def roberval_score(self) -> int:
        return self.ai_score

    @property
    def roberval_shots_count(self) -> int:
        return self.ai_shots_count


class GameStateMachine:
    """Manages the full lifecycle of an ITTF ping pong match against Roberval."""

    OPPONENT_NAME: str = "ROBERVAL"

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
        self.announcement: str = "NOVA PARTIDA CONTRA ROBERVAL! FECHE A MAO ✊ PARA INICIAR"
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
        self.announcement = "NOVA PARTIDA CONTRA ROBERVAL! FECHE A MAO ✊ PARA INICIAR"
        self.announcement_timer = 3.5
        self._shots_speeds.clear()
        self.physics.reset_serve(self.ball, server="PLAYER")

    def prepare_serve(self):
        """Position ball floating ready for serve."""
        self.physics.reset_serve(self.ball, server=self.server)
        self.status = MatchStatus.SERVING
        self.current_rally_count = 0
        if self.server == "PLAYER":
            self.announcement = "BOLA PRONTA! SACAR COM O MOVIMENTO DO BRACO! 🏓"
            self.announcement_timer = 3.0
        else:
            self.announcement = "SAQUE DO ROBERVAL..."
            self.announcement_timer = 1.2

    def strike_serve(
        self,
        swing_speed: float = 2.2,
        vx_offset: float = 0.0,
        vy_lift: float = 0.0,
    ):
        """Hit the serve with launch dynamics based on player's physical hand/arm swing."""
        if self.status in (MatchStatus.READY, MatchStatus.SERVING):
            self.status = MatchStatus.RALLY

            # Clamped swing speed in m/s (from 0.8 to 6.0 m/s)
            clamped_speed = max(0.8, min(6.0, swing_speed))

            # Forward velocity scales with arm swing speed (approx 2.4 to 5.6 m/s)
            forward_vz = 2.2 + clamped_speed * 0.95

            # Vertical arc ensures clearing net comfortably, with bonus from upward arm lift
            vertical_vy = max(1.4, min(2.7, 1.60 + max(0.0, vy_lift) * 0.35))

            # Horizontal placement towards corners
            horizontal_vx = max(-1.25, min(1.25, vx_offset * 1.9))

            self.ball.x = self.player_paddle.x * 0.5
            self.ball.y = 0.28
            self.ball.z = -1.35
            self.ball.vx = horizontal_vx
            self.ball.vy = vertical_vy
            self.ball.vz = forward_vz
            self.ball.in_play = True
            self.ball.is_serve = True
            self.ball.last_hitter = "PLAYER"
            self.ball.spin_y = horizontal_vx * 6.0
            self.ball.spin_x = vy_lift * 6.0

            actual_speed_kmh = self.ball.speed_kmh
            shot_desc = "POWER SERVE" if actual_speed_kmh >= 30.0 else "SAQUE"

            self.announcement = f"{shot_desc}! {actual_speed_kmh:.0f} km/h"
            self.announcement_timer = 1.2

            # Log serve shot
            shot = ShotRecord(
                shot_id=str(uuid.uuid4()),
                match_id=self.match_id,
                shot_number=1,
                hitter="PLAYER",
                shot_type="SERVE",
                speed_kmh=actual_speed_kmh,
                spin="TOPSPIN" if vy_lift > 0.3 else "NONE",
            )
            self.current_shot = shot
            self.stats.shots.append(shot)
            self._shots_speeds.append(actual_speed_kmh)

    def trigger_serve(self):
        """Legacy / fallback programmatic serve execution."""
        if self.status in (MatchStatus.READY, MatchStatus.SERVING):
            if self.server == "PLAYER":
                self.strike_serve(swing_speed=2.2)
            else:
                self._execute_roberval_serve()

    def _execute_roberval_serve(self):
        """Execute Roberval's serve towards player."""
        self.status = MatchStatus.RALLY
        self.ball.x = -0.15
        self.ball.y = 0.28
        self.ball.z = 1.35
        self.ball.vx = -0.2 if self.ai.rng.random() < self.ai.difficulty.corner_aim_tendency else 0.2
        self.ball.vy = 1.80
        self.ball.vz = -2.9
        self.ball.in_play = True
        self.ball.is_serve = True
        self.ball.last_hitter = "ROBERVAL"
        self.announcement = "SAQUE DO ROBERVAL!"
        self.announcement_timer = 1.0

        shot = ShotRecord(
            shot_id=str(uuid.uuid4()),
            match_id=self.match_id,
            shot_number=1,
            hitter="ROBERVAL",
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
            # Auto serve for Roberval
            if self.server in ("AI", "ROBERVAL"):
                if self.announcement_timer <= 0:
                    self._execute_roberval_serve()
            elif self.server == "PLAYER":
                # Ball hovers gently at baseline waiting for player's physical swing
                self.ball.x = self.player_paddle.x * 0.5
                self.ball.y = 0.28 + 0.02 * math.sin(time.time() * 5.0)
                self.ball.z = -1.35
                self.ball.vx = 0.0
                self.ball.vy = 0.0
                self.ball.vz = 0.0
                self.ball.in_play = True
                self.ball.is_serve = True
            return

        if self.status == MatchStatus.RALLY:
            # Update Roberval paddle behavior
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

            hitter_name = "PLAYER" if event.hitter == "PLAYER" else "ROBERVAL"
            shot = ShotRecord(
                shot_id=str(uuid.uuid4()),
                match_id=self.match_id,
                shot_number=self.current_rally_count,
                hitter=hitter_name,
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

            # Table bounce handling: In arcade play, serve simply lands on opponent's table
            if self.ball.is_serve:
                opponent_side = "ROBERVAL" if self.server == "PLAYER" else "PLAYER"
                if event.details in (opponent_side, "AI"):
                    # Successful serve: Open rally begins!
                    self.ball.is_serve = False
                    self.ball.bounce_count_current_hit = 1
                elif event.details == ("PLAYER" if self.server == "PLAYER" else "ROBERVAL"):
                    # Bounced on own side first (classical serve)
                    pass
            else:
                # Regular rally: if ball bounces twice on the same side without being hit -> Point to opponent!
                if self.ball.bounce_count_current_hit >= 2:
                    self.ball.in_play = False
                    scoring_party = "PLAYER" if event.details in ("AI", "ROBERVAL") else "ROBERVAL"
                    self._award_point(scoring_party, "DOIS QUIQUES!")

        elif event.event_type == "NET_FAULT":
            self.ball.in_play = False
            hitter = event.hitter or self.server
            if hitter == "PLAYER":
                self._award_point("ROBERVAL", "BOLA NA REDE! PONTO DO ROBERVAL")
            else:
                self._award_point("PLAYER", "ROBERVAL NA REDE! PONTO SEU!")

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
                self._award_point("ROBERVAL", "FORA! PONTO DO ROBERVAL")
        else:
            if self.ball.bounced_player_side:
                self._award_point("ROBERVAL", "WINNER POINT DO ROBERVAL!")
            else:
                self._award_point("PLAYER", "ROBERVAL ERROU! PONTO SEU!")

    def _award_point(self, winner: str, reason: str):
        """Add point, check game over, update serve rotation."""
        self.status = MatchStatus.POINT_SCORED
        self.stats.total_rallies += 1

        if winner == "PLAYER":
            self.player_score += 1
            self.announcement = f"PONTO SEU! {reason}"
        else:
            self.ai_score += 1
            self.announcement = f"PONTO DO ROBERVAL! {reason}"

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
            match_winner = "PLAYER" if p_score > a_score else "ROBERVAL"
            self.stats.winner = match_winner
            winner_display = "VOCE VENCEU" if match_winner == "PLAYER" else "ROBERVAL VENCEU"
            self.announcement = f"FIM DE JOGO! {winner_display} ({p_score} - {a_score})!"
            self.announcement_timer = 5.0
            return

        # Check deuce
        if p_score >= 10 and a_score >= 10:
            if p_score == a_score:
                self.announcement += " [DEUCE / EMPATE]"
            elif p_score > a_score:
                self.announcement += " [VANTAGEM VOCE]"
            else:
                self.announcement += " [VANTAGEM ROBERVAL]"

        # Update serve rotation (every 2 points, or every 1 point in deuce)
        total_points = p_score + a_score
        if p_score >= 10 and a_score >= 10:
            # In deuce, serve alternates every single point
            self.server = "ROBERVAL" if total_points % 2 == 1 else "PLAYER"
        else:
            # Alternates every 2 points
            rotation_idx = (total_points // self.rules.SERVES_PER_ROTATION) % 2
            self.server = "PLAYER" if rotation_idx == 0 else "ROBERVAL"

        # Reset for next serve
        self.prepare_serve()
        if self.server == "PLAYER":
            self.announcement += " | SEU SAQUE (GOLPE COM O BRACO 🏓)"
        else:
            self.announcement += " | SAQUE DO ROBERVAL"
            self.announcement_timer = 2.0
