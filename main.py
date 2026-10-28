"""AI Ping Pong - Standalone Desktop Runner (60 FPS Native OpenCV Window with Mouse & Webcam)."""

import sys
import time
import cv2
import numpy as np

from src.config import CameraConfig, DisplayConfig
from src.engine.game_state import GameStateMachine, MatchStatus
from src.ui.renderer_3d import PerspectiveRenderer3D
from src.ui.sound_effects import SoundEffectsManager
from src.vision.camera import WebcamSource, SyntheticCameraSource
from src.vision.tracker import HandTracker
from src.databricks.storage import get_storage


def run_desktop_app():
    """Run standalone 60 FPS desktop table tennis game with dual mouse/webcam controls."""
    print("=" * 70)
    print("🏓 AI PING PONG - MODO ARCADE WII SPORTS (60 FPS NATIVO)")
    print("=" * 70)
    print("Controles Intuitivos:")
    print("  🖱️  MOUSE          : Mova o cursor na janela para mover a raquete")
    print("  🖱️  CLIQUE ESQUERDO: Sacar bola / Power Smash")
    print("  ✋ WEBCAM         : Mova sua mão na câmera para controlar a raquete")
    print("  [ESPAÇO]          : Sacar a bola")
    print("  [R]               : Reiniciar partida")
    print("  [1 - 4]           : Mudar dificuldade (1:Easy, 2:Medium, 3:Hard, 4:Champ)")
    print("  [S]               : Salvar no Databricks / Local")
    print("  [Q/ESC]           : Sair do Jogo")
    print("=" * 70)

    # Initialize components
    storage = get_storage()
    game = GameStateMachine(difficulty="MEDIUM")
    renderer = PerspectiveRenderer3D()
    tracker = HandTracker()
    sounds = SoundEffectsManager()

    # Mouse state
    mouse_pos = [0.0, 0.20]  # [world_x, world_y]
    mouse_last_moved = [0.0]
    click_swing = [False]

    def on_mouse_event(event, x, y, flags, param):
        # Canvas dimensions: 800 x 560
        # Map mouse X: [0, 800] -> world_x [-0.90, +0.90]
        mouse_pos[0] = ((x / 800.0) - 0.5) * 1.80
        # Map mouse Y: [0, 560] -> world_y [0.05, 0.65] (inverted)
        mouse_pos[1] = max(0.05, min(0.65, (1.0 - (y / 560.0)) * 0.70))
        mouse_last_moved[0] = time.time()

        if event == cv2.EVENT_LBUTTONDOWN:
            click_swing[0] = True

    # Try opening webcam, fallback to synthetic camera
    cam = WebcamSource(device_index=0, width=640, height=480)
    if not cam.is_opened:
        print("[AVISO] Webcam física não detectada. Ativando Câmera Sintética...")
        cam = SyntheticCameraSource(width=640, height=480)

    window_name = "AI Ping Pong - Wii Sports Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse_event)

    prev_time = time.time()

    try:
        while True:
            now = time.time()
            dt = max(1e-4, min(0.05, now - prev_time))
            prev_time = now

            # 1. Read camera frame
            ret, frame = cam.read()
            if not ret or frame is None:
                break

            # 2. Track hand via MediaPipe
            track_res = tracker.process_frame(frame)

            # Determine paddle position: Mouse takes priority if actively moved recently, else hand
            use_mouse = (now - mouse_last_moved[0]) < 1.2
            if use_mouse or not track_res.detected:
                target_x = mouse_pos[0]
                target_y = mouse_pos[1]
                target_z = -1.65
            else:
                target_x = track_res.world_x
                target_y = track_res.world_y
                target_z = track_res.world_z

            game.player_paddle.update_position(target_x, target_y, target_z, dt)

            # 3. Handle Mouse Click / Swing
            if click_swing[0]:
                click_swing[0] = False
                if game.status in (MatchStatus.READY, MatchStatus.SERVING):
                    game.trigger_serve()
                elif game.status == MatchStatus.RALLY:
                    # Apply power smash forward impulse
                    game.player_paddle.vx = (game.ball.x - game.player_paddle.x) * 4.0
                    game.player_paddle.vy = -1.5
                    game.player_paddle.vz = 8.0

            # 4. Step game simulation
            prev_rally = game.current_rally_count
            prev_bounces = game.ball.bounce_count_current_hit

            game.update(dt=dt)

            if game.current_rally_count > prev_rally:
                sounds.trigger_hit_spark(400, 320, "DRIVE")
            elif game.ball.bounce_count_current_hit > prev_bounces:
                sounds.play_bounce()

            # 5. Render 3D Perspective game world
            scene_frame = renderer.render_frame(game)
            sounds.update_and_render(scene_frame, dt)

            # 6. Composite Camera HUD (Bottom Right Corner)
            cam_view = cv2.resize(track_res.annotated_frame, (260, 195))
            sh, sw = scene_frame.shape[:2]
            ch, cw = cam_view.shape[:2]
            scene_frame[sh - ch - 15 : sh - 15, sw - cw - 15 : sw - 15] = cam_view
            cv2.rectangle(
                scene_frame,
                (sw - cw - 15, sh - ch - 15),
                (sw - 15, sh - 15),
                (255, 255, 255),
                2,
            )

            # 7. Render Controls Legend on Bottom Left
            control_hint = "CONTROLE: MOUSE / WEBCAM | CLIQUE/ESPACO: SACAR | R: RESET"
            cv2.putText(
                scene_frame,
                control_hint,
                (25, sh - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                2,
            )

            cv2.imshow(window_name, scene_frame)

            # 8. Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key == 32:  # Space
                game.trigger_serve()
            elif key in (ord("r"), ord("R")):
                if game.stats.shots:
                    storage.save_match(game.stats)
                game.start_new_match()
            elif key in (ord("s"), ord("S")):
                storage.save_match(game.stats)
                print(f"[DATABRICKS] Partida {game.match_id} salva.")
            elif key == ord("1"):
                game.start_new_match("EASY")
            elif key == ord("2"):
                game.start_new_match("MEDIUM")
            elif key == ord("3"):
                game.start_new_match("HARD")
            elif key == ord("4"):
                game.start_new_match("CHAMPION")

    finally:
        if game.stats.shots:
            storage.save_match(game.stats)
            print(f"[TELEMETRIA] Partida {game.match_id} registrada.")

        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_desktop_app()
