"""AI Ping Pong - Standalone Desktop Runner (60 FPS Native OpenCV Window)."""

import sys
import time
import cv2
import numpy as np

from src.config import CameraConfig, DisplayConfig
from src.engine.game_state import GameStateMachine, MatchStatus
from src.ui.renderer_3d import PerspectiveRenderer3D
from src.vision.camera import WebcamSource, SyntheticCameraSource
from src.vision.tracker import HandTracker
from src.databricks.storage import get_storage


def run_desktop_app():
    """Run standalone 60 FPS desktop table tennis game."""
    print("=" * 65)
    print("🏓 AI PING PONG - DESKTOP 60 FPS MODE (WII SPORTS INSPIRED)")
    print("=" * 65)
    print("Teclas de Controle:")
    print("  [ESPAÇO] : Sacar a bola")
    print("  [R]      : Reiniciar / Nova Partida")
    print("  [1 - 4]  : Mudar Dificuldade (1:Easy, 2:Medium, 3:Hard, 4:Champ)")
    print("  [S]      : Salvar Estatísticas no Databricks / Local")
    print("  [Q/ESC]  : Sair do Jogo")
    print("=" * 65)

    # Initialize components
    storage = get_storage()
    game = GameStateMachine(difficulty="MEDIUM")
    renderer = PerspectiveRenderer3D()
    tracker = HandTracker()

    # Try opening webcam, fallback to synthetic camera if not present
    cam = WebcamSource(device_index=0, width=640, height=480)
    if not cam.is_opened:
        print("[AVISO] Nenhuma webcam detectada no índice 0. Ativando Câmera Sintética...")
        cam = SyntheticCameraSource(width=640, height=480)

    window_name = "AI Ping Pong - Wii Sports Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

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
            if track_res.detected:
                game.player_paddle.update_position(
                    track_res.world_x, track_res.world_y, track_res.world_z, dt
                )

            # 3. Step game simulation
            game.update(dt=dt)

            # 4. Render 3D Perspective game world
            scene_frame = renderer.render_frame(game)

            # 5. Composite Dual-Pane Display (3D Game on Left, Camera on Right)
            cam_view = cv2.resize(track_res.annotated_frame, (320, 240))
            # Overlay webcam in bottom-right corner of scene
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

            cv2.imshow(window_name, scene_frame)

            # 6. Keyboard handling
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
                print(f"[DATABRICKS] Partida {game.match_id} salva com sucesso!")
            elif key == ord("1"):
                game.start_new_match("EASY")
            elif key == ord("2"):
                game.start_new_match("MEDIUM")
            elif key == ord("3"):
                game.start_new_match("HARD")
            elif key == ord("4"):
                game.start_new_match("CHAMPION")

    finally:
        # Save telemetry on exit if played
        if game.stats.shots:
            storage.save_match(game.stats)
            print(f"[TELEMETRIA] Partida {game.match_id} registrada.")

        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_desktop_app()
