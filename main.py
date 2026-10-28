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
    prev_fist = False

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

            # 2.1. Hand Fist Gesture (Fechar a Mão) ONLY to Start Game / Prepare Serve
            current_fist = track_res.detected and track_res.is_fist
            if current_fist and not prev_fist:
                if game.status == MatchStatus.READY:
                    game.prepare_serve()
                    sounds.play_bounce()
                    print("[GESTO] Mão fechada ✊: Jogo iniciado! Bola posicionada para o saque.")
                elif game.status == MatchStatus.GAME_OVER:
                    game.start_new_match()
                    print("[GESTO] Mão fechada ✊: Nova partida iniciada contra Roberval!")
            prev_fist = current_fist

            # Determine paddle position: Mouse takes priority if actively moved recently, else hand
            use_mouse = (now - mouse_last_moved[0]) < 1.2
            if use_mouse or not track_res.detected:
                target_x = mouse_pos[0]
                target_y = mouse_pos[1]
                target_z = -1.40
            else:
                target_x = track_res.world_x
                target_y = track_res.world_y
                target_z = -1.40

            game.player_paddle.update_position(target_x, target_y, target_z, dt)

            # 2.2. SERVE BY PHYSICAL HAND/ARM SWING MOTION & FORCE
            if game.status == MatchStatus.SERVING and game.server == "PLAYER":
                arm_speed = game.player_paddle.speed if not track_res.detected else max(
                    getattr(track_res, "hook_power", game.player_paddle.speed),
                    game.player_paddle.speed,
                )
                is_swing = getattr(track_res, "is_hook_swing", False) or (arm_speed >= 0.85) or (game.player_paddle.vz > 0.8)

                dx = game.ball.x - game.player_paddle.x
                dy = game.ball.y - game.player_paddle.y
                dist_xy = math.sqrt(dx * dx + dy * dy)

                if (is_swing and dist_xy <= 0.65) or (dist_xy <= 0.35 and arm_speed >= 0.70):
                    vy_lift = game.player_paddle.vy if game.player_paddle.vy > 0 else 0.5
                    game.strike_serve(
                        swing_speed=arm_speed,
                        vx_offset=(game.ball.x - game.player_paddle.x),
                        vy_lift=vy_lift,
                    )
                    sounds.play_hit(is_smash=(arm_speed >= 2.4))
                    sounds.trigger_hit_spark(450, 480, "SERVE")
                    print(f"[SAQUE] Golpe executado com forca do braco: {arm_speed:.2f} m/s ({game.ball.speed_kmh:.1f} km/h)")

            # 3. Handle Mouse Click / Swing
            if click_swing[0]:
                click_swing[0] = False
                if game.status == MatchStatus.READY:
                    game.prepare_serve()
                    sounds.play_bounce()
                elif game.status == MatchStatus.SERVING and game.server == "PLAYER":
                    arm_speed = max(2.4, game.player_paddle.speed)
                    game.strike_serve(swing_speed=arm_speed, vx_offset=(game.ball.x - game.player_paddle.x), vy_lift=0.6)
                    sounds.play_hit(is_smash=True)
                    sounds.trigger_hit_spark(450, 480, "SERVE")
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
                shot_type = game.current_shot.shot_type if game.current_shot else "DRIVE"
                sounds.trigger_hit_spark(400, 320, shot_type)
            elif game.ball.bounce_count_current_hit > prev_bounces:
                sounds.play_bounce()

            # 5. Render 3D Perspective game world
            scene_frame = renderer.render_frame(game)
            sounds.update_and_render(scene_frame, dt)

            # 6. Composite Camera Picture-in-Picture (Top-Right: completely outside table area)
            cw, ch = 190, 142
            cam_view = cv2.resize(track_res.annotated_frame, (cw, ch))
            sh, sw = scene_frame.shape[:2]
            cam_x1 = sw - cw - 15
            cam_y1 = 82
            cam_x2 = sw - 15
            cam_y2 = cam_y1 + ch

            scene_frame[cam_y1:cam_y2, cam_x1:cam_x2] = cam_view
            cv2.rectangle(
                scene_frame,
                (cam_x1, cam_y1),
                (cam_x2, cam_y2),
                (255, 255, 255),
                2,
            )
            # Small discrete badge for camera
            cv2.rectangle(scene_frame, (cam_x1, cam_y1 - 16), (cam_x1 + 72, cam_y1), (25, 25, 30), -1)
            cv2.putText(scene_frame, "WEBCAM", (cam_x1 + 6, cam_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 255), 1)

            # 7. Render Discrete Controls Legend on Bottom Left
            control_hint = "INICIAR: FECHE A MAO ✊ | SACAR: GOLPE COM O BRACO 🏓 | ADVERSARIO: ROBERVAL"
            cv2.putText(
                scene_frame,
                control_hint,
                (20, sh - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (220, 220, 220),
                1,
            )

            cv2.imshow(window_name, scene_frame)

            # 8. Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key == 32:  # Space
                if game.status == MatchStatus.READY:
                    game.prepare_serve()
                    sounds.play_bounce()
                elif game.status == MatchStatus.SERVING and game.server == "PLAYER":
                    game.strike_serve(swing_speed=2.5, vx_offset=0.0, vy_lift=0.6)
                    sounds.play_hit(is_smash=True)
                    sounds.trigger_hit_spark(450, 480, "SERVE")
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
