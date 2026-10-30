"""User movement calibration and personalized gesture training suite for table tennis."""

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class UserGestureProfile:
    """User-specific motion thresholds and biomechanical parameters."""
    handedness: str = "RIGHT"              # "RIGHT" or "LEFT"
    serve_speed_threshold: float = 0.85    # m/s to trigger serve
    forehand_speed_threshold: float = 1.05 # m/s for deliberate forehand
    backhand_speed_threshold: float = 1.05 # m/s for deliberate backhand
    smash_speed_threshold: float = 2.40    # m/s for power smash
    arm_power_multiplier: float = 1.0      # power scaling
    is_calibrated: bool = False
    calibrated_at: str = ""

    def save(self, filepath: str = "user_calibration.json"):
        """Save calibrated profile to JSON."""
        self.calibrated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str = "user_calibration.json") -> "UserGestureProfile":
        """Load profile from JSON, or return sensible defaults if not found."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return cls(**data)
            except Exception:
                pass
        return cls()


class GestureTrainer:
    """Interactive step-by-step training and calibration wizard for arm and hand movements."""

    STEPS = [
        ("REST", "1. POSICAO DE ESPERA: Mantenha o braco preparado na frente"),
        ("SERVE", "2. TREINO DE SAQUE: Execute 2 movimentos de Saque com o braco"),
        ("FOREHAND", "3. GOLPE DE DIREITA: Execute 2 batidas de Forehand (para dentro)"),
        ("BACKHAND", "4. GOLPE DE ESQUERDA: Execute 2 batidas de Backhand (para fora)"),
        ("SMASH", "5. CORTADA POTENTE: Execute 2 smashes fortes de cima para baixo!"),
    ]

    def __init__(self, profile: Optional[UserGestureProfile] = None):
        self.profile = profile or UserGestureProfile.load()
        self.is_active: bool = False
        self.current_step_idx: int = 0
        self.step_start_time: float = 0.0

        # Samples collected per step
        self.samples: Dict[str, List[float]] = {
            "SERVE": [],
            "FOREHAND": [],
            "BACKHAND": [],
            "SMASH": [],
        }

        self.last_swing_time: float = 0.0
        self.has_returned_to_rest: bool = True
        self.just_recorded_rep: bool = False
        self.feedback_msg: str = ""
        self.feedback_color: Tuple[int, int, int] = (0, 230, 255)
        self.feedback_timer: float = 0.0

    def start(self):
        """Begin interactive gesture calibration."""
        self.is_active = True
        self.current_step_idx = 0
        self.step_start_time = time.time()
        self.has_returned_to_rest = True
        self.just_recorded_rep = False
        self.last_swing_time = 0.0
        for k in self.samples:
            self.samples[k].clear()
        self.feedback_msg = "INICIANDO CALIBRACAO DE BRACO..."
        self.feedback_color = (0, 230, 255)
        self.feedback_timer = 2.0

    def stop(self):
        """Cancel gesture calibration and immediately hide overlay."""
        self.is_active = False
        self.feedback_timer = 0.0
        self.feedback_msg = ""

    @property
    def current_step(self) -> Tuple[str, str]:
        if 0 <= self.current_step_idx < len(self.STEPS):
            return self.STEPS[self.current_step_idx]
        return ("COMPLETE", "CALIBRACAO CONCLUIDA COM SUCESSO!")

    def update(self, speed: float, vx: float, vy: float, is_fist: bool, dt: float) -> bool:
        """
        Process motion sample during calibration.
        Returns True when calibration is fully completed.
        """
        if self.feedback_timer > 0:
            self.feedback_timer = max(0.0, self.feedback_timer - dt)

        if not self.is_active:
            return False

        now = time.time()
        step_key, _ = self.current_step

        # Step 0: REST (Hold position for 2.0 seconds)
        if step_key == "REST":
            if now - self.step_start_time >= 2.0:
                self._advance_step("Posicao de espera calibrada! Avancando para o Saque...")
            return False

        # Steps with active swings (need 2 distinct reps with return to rest)
        cooldown = now - self.last_swing_time
        target_reps = 2

        # Check if arm returned to rest before allowing the next rep
        if not self.has_returned_to_rest:
            if speed < 0.28 or cooldown >= 0.85:
                self.has_returned_to_rest = True
                rep_count = len(self.samples.get(step_key, []))
                if rep_count == 1:
                    self.feedback_msg = "OTIMO! AGORA EXECUTE O 2o MOVIMENTO..."
                    self.feedback_color = (0, 230, 255)
                    self.feedback_timer = 2.0

        can_trigger = self.has_returned_to_rest and (cooldown >= 0.70 or self.last_swing_time == 0.0)

        if step_key == "SERVE":
            # Intentional serve stroke: upward vy > 0.12 or general forward swing with speed >= 0.75
            if can_trigger and speed >= 0.75 and (vy > 0.12 or abs(vx) > 0.25):
                self.samples["SERVE"].append(speed)
                self.last_swing_time = now
                self.has_returned_to_rest = False
                self.just_recorded_rep = True
                rep = len(self.samples["SERVE"])
                if rep < target_reps:
                    self.feedback_msg = f"1o SAQUE REGISTRADO! ({speed * 3.6:.0f} km/h) -> Volte o braco e faca o 2o"
                    self.feedback_color = (0, 255, 120)
                    self.feedback_timer = 2.0
                else:
                    self.feedback_msg = f"2o SAQUE REGISTRADO! ({speed * 3.6:.0f} km/h) -> Saque Calibrado!"
                    self.feedback_color = (0, 255, 120)
                    self.feedback_timer = 1.0
                    self._advance_step("Saque calibrado! Agora treine o Forehand...")

        elif step_key == "FOREHAND":
            # Right-handed: positive vx or forward motion
            is_right = (self.profile.handedness == "RIGHT")
            is_fh = (vx > 0.20) if is_right else (vx < -0.20)
            if can_trigger and speed >= 0.75 and is_fh:
                self.samples["FOREHAND"].append(speed)
                self.last_swing_time = now
                self.has_returned_to_rest = False
                self.just_recorded_rep = True
                rep = len(self.samples["FOREHAND"])
                if rep < target_reps:
                    self.feedback_msg = f"1o FOREHAND REGISTRADO! ({speed * 3.6:.0f} km/h) -> Volte o braco e faca o 2o"
                    self.feedback_color = (0, 230, 255)
                    self.feedback_timer = 2.0
                else:
                    self.feedback_msg = f"2o FOREHAND REGISTRADO! ({speed * 3.6:.0f} km/h) -> Forehand Calibrado!"
                    self.feedback_color = (0, 255, 120)
                    self.feedback_timer = 1.0
                    self._advance_step("Forehand calibrado! Agora treine o Backhand...")

        elif step_key == "BACKHAND":
            is_right = (self.profile.handedness == "RIGHT")
            is_bh = (vx < -0.20) if is_right else (vx > 0.20)
            if can_trigger and speed >= 0.75 and is_bh:
                self.samples["BACKHAND"].append(speed)
                self.last_swing_time = now
                self.has_returned_to_rest = False
                self.just_recorded_rep = True
                rep = len(self.samples["BACKHAND"])
                if rep < target_reps:
                    self.feedback_msg = f"1o BACKHAND REGISTRADO! ({speed * 3.6:.0f} km/h) -> Volte o braco e faca o 2o"
                    self.feedback_color = (255, 120, 255)
                    self.feedback_timer = 2.0
                else:
                    self.feedback_msg = f"2o BACKHAND REGISTRADO! ({speed * 3.6:.0f} km/h) -> Backhand Calibrado!"
                    self.feedback_color = (0, 255, 120)
                    self.feedback_timer = 1.0
                    self._advance_step("Backhand calibrado! Agora execute cortadas potentes (Smash)...")

        elif step_key == "SMASH":
            if can_trigger and speed >= 1.25 and (vy < -0.15 or speed >= 1.6):
                self.samples["SMASH"].append(speed)
                self.last_swing_time = now
                self.has_returned_to_rest = False
                self.just_recorded_rep = True
                rep = len(self.samples["SMASH"])
                if rep < target_reps:
                    self.feedback_msg = f"💥 1o SMASH REGISTRADO! ({speed * 3.6:.0f} km/h) -> Prepare o 2o Smash!"
                    self.feedback_color = (0, 140, 255)
                    self.feedback_timer = 2.0
                else:
                    self.feedback_msg = f"💥 2o SMASH REGISTRADO! ({speed * 3.6:.0f} km/h) -> Calibracao Concluida!"
                    self.feedback_color = (0, 255, 120)
                    self.feedback_timer = 1.5
                    self._finalize_calibration()
                    return True

        return False

    def skip_step(self):
        """Skip current calibration step."""
        if self.current_step_idx < len(self.STEPS) - 1:
            self._advance_step("Etapa pulada pelo jogador.")
        else:
            self._finalize_calibration()

    def _advance_step(self, message: str):
        self.current_step_idx += 1
        self.step_start_time = time.time()
        self.feedback_msg = message
        self.feedback_color = (0, 255, 120)
        self.feedback_timer = 1.5
        self.has_returned_to_rest = True
        self.last_swing_time = 0.0
        self.step_start_time = time.time()
        self.feedback_msg = message
        self.feedback_color = (0, 255, 120)
        self.feedback_timer = 1.5

    def _finalize_calibration(self):
        """Compute personalized thresholds from captured samples."""
        self.is_active = False

        if self.samples["SERVE"]:
            avg_serve = float(np.mean(self.samples["SERVE"]))
            self.profile.serve_speed_threshold = max(0.40, min(1.50, avg_serve * 0.70))

        if self.samples["FOREHAND"]:
            avg_fh = float(np.mean(self.samples["FOREHAND"]))
            self.profile.forehand_speed_threshold = max(0.45, min(2.0, avg_fh * 0.75))

        if self.samples["BACKHAND"]:
            avg_bh = float(np.mean(self.samples["BACKHAND"]))
            self.profile.backhand_speed_threshold = max(0.45, min(2.0, avg_bh * 0.75))

        if self.samples["SMASH"]:
            avg_smash = float(np.mean(self.samples["SMASH"]))
            self.profile.smash_speed_threshold = max(1.20, min(4.5, avg_smash * 0.80))

        self.profile.is_calibrated = True
        self.profile.save()
        self.feedback_msg = "PERFIL BIOMECANICO SALVO COM SUCESSO! SEUS MOVIMENTOS FORAM APRENDIDOS."
        self.feedback_color = (0, 255, 120)
        self.feedback_timer = 1.5

    def render_overlay(self, frame: np.ndarray, current_speed_kmh: float = 0.0):
        """Render high-tech sports training HUD overlay on frame."""
        h, w = frame.shape[:2]
        cx = int(w / 2)

        # 1. Dark high-contrast card
        card_w, card_h = 760, 240
        x1 = cx - int(card_w / 2)
        y1 = 120
        x2 = x1 + card_w
        y2 = y1 + card_h

        # Solid background to cleanly obscure scene/menu behind
        cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 20, 28), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 5), (0, 230, 255), -1)
        cv2.line(frame, (x1 + 20, y1 + 45), (x2 - 20, y1 + 45), (35, 45, 60), 1)

        # 2. Title & Step Indicator
        title = "MODO DE CALIBRACAO E TREINAMENTO DE BRACO"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.putText(frame, title, (cx - int(tw / 2), y1 + 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        step_key, step_instruction = self.current_step
        total_steps = len(self.STEPS)
        step_progress_str = f"ETAPA {min(total_steps, self.current_step_idx + 1)} / {total_steps}"
        cv2.putText(frame, step_progress_str, (x1 + 25, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

        # 3. Main Instruction
        cv2.putText(frame, step_instruction, (x1 + 25, y1 + 98), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 240, 200), 1, cv2.LINE_AA)

        # 4. Progress Bars for Reps
        if step_key in self.samples:
            reps = len(self.samples[step_key])
            if reps == 0:
                hint_str = "-> Faca o 1o movimento"
            elif reps == 1:
                hint_str = "-> Volte o braco e faca o 2o movimento"
            else:
                hint_str = "-> Concluido!"
            rep_text = f"Movimentos: {reps} / 2  {hint_str}"
            cv2.putText(frame, rep_text, (x1 + 25, y1 + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 230, 255), 1, cv2.LINE_AA)

            # Draw 2 rep indicator boxes
            for i in range(2):
                bx = x1 + 450 + i * 45
                by = y1 + 122
                bw, bh = 36, 20
                is_done = (i < reps)
                box_color = (0, 255, 120) if is_done else (35, 45, 60)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, -1)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 230, 255) if is_done else (80, 100, 130), 1)
                lbl = "OK" if is_done else f"{i+1}o"
                cv2.putText(frame, lbl, (bx + 6, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (15, 25, 20) if is_done else (160, 180, 200), 1)

        # 5. Live Speedometer
        speed_str = f"VELOCIDADE DO BRACO: {current_speed_kmh:.1f} KM/H"
        spd_color = (0, 140, 255) if current_speed_kmh > 20 else ((0, 230, 255) if current_speed_kmh > 8 else (180, 200, 220))
        cv2.putText(frame, speed_str, (x1 + 25, y1 + 175), cv2.FONT_HERSHEY_DUPLEX, 0.48, spd_color, 1, cv2.LINE_AA)

        # 6. Transient Feedback Toast
        if self.feedback_timer > 0 and self.feedback_msg:
            cv2.putText(frame, self.feedback_msg, (x1 + 25, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.feedback_color, 1, cv2.LINE_AA)
        else:
            footer = "[ESPACO] Pular Etapa   |   [T / ESC] Sair e Salvar"
            cv2.putText(frame, footer, (x1 + 25, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 160, 180), 1, cv2.LINE_AA)
