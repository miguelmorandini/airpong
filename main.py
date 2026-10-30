"""AI Ping Pong - Standalone Desktop Runner (60 FPS Native OpenCV Window with Mouse & Webcam)."""

import math
import sys
import time
import cv2
import numpy as np

from src.config import CameraConfig, DisplayConfig
from src.engine.game_state import GameStateMachine, MatchStatus
from src.ui.renderer_3d import PerspectiveRenderer3D
from src.ui.sound_effects import SoundEffectsManager
from src.vision.camera import WebcamSource, SyntheticCameraSource
from src.vision.calibration import GestureTrainer, UserGestureProfile
from src.vision.tracker import HandTracker
from src.databricks.storage import get_storage


def run_desktop_app():
    """Run standalone 60 FPS desktop table tennis game with dual mouse/webcam controls."""
    print("=" * 70)
    print("🏓 AI PING PONG - MODO ARCADE WII SPORTS (60 FPS NATIVO)")
    print("=" * 70)
    print("Controles Intuitivos:")
    print("  🏓  EMPUNHADURA   : Segure a mão fechada como se estivesse segurando a raquete!")
    print("                      Suporta Clássico (Shakehand), Caneta Japonesa e Classineta.")
    print("  ✋  WEBCAM        : Mova a mão fechada e o braço para rebater, sacar e dar efeito")
    print("  🖱️  MOUSE (ALT)   : Mova o cursor na janela caso prefira controle por mouse")
    print("  🖱️  CLIQUE ESQ.   : Interagir no Menu / Sacar bola / Power Smash")
    print("  [M]               : Abrir / Fechar Menu de Lobby e Pausa")
    print("  [H]               : Alternar Braço Dominante (Auto / Destro / Canhoto)")
    print("  [ESPAÇO]          : Iniciar Partida / Sacar a bola")
    print("  [T]               : Calibrar / Treinar Movimentos do Braço")
    print("  [F]               : Alternar Tela Cheia / Maximizado")
    print("  [R]               : Reiniciar partida")
    print("  [1 - 3]           : Mudar dificuldade (1:Fácil, 2:Médio, 3:Difícil)")
    print("  [S]               : Salvar no Databricks / Local")
    print("  [Q/ESC]           : Sair do Jogo")
    print("=" * 70)

    # Initialize components
    storage = get_storage()
    game = GameStateMachine(difficulty="MEDIUM")
    renderer = PerspectiveRenderer3D()
    tracker = HandTracker()
    sounds = SoundEffectsManager()

    # User Profile & Gesture Training
    user_profile = UserGestureProfile.load()
    tracker.gesture_classifier.apply_profile(user_profile)
    trainer = GestureTrainer(profile=user_profile)
    if user_profile.is_calibrated:
        print(f"[PERFIL] Calibração de movimentos carregada! Calibrado em: {user_profile.calibrated_at}")

    # Mouse & Input state (input-systems skill)
    mouse_pos = [0.0, 0.20]  # [world_x, world_y]
    mouse_screen = [0, 0]    # [screen_x, screen_y] for menu hover
    mouse_last_moved = [0.0]
    click_swing = [False]
    last_click_pos = [(0, 0)]
    in_menu = True            # Pre-game Start Menu / Lobby active initially
    click_buffer_timer = 0.0  # 160ms input buffer for responsive smashes
    hit_stop_timer = 0.0      # 40ms freeze-frame micro-pause on smashes (game-feel)

    def on_mouse_event(event, x, y, flags, param):
        mouse_screen[0] = x
        mouse_screen[1] = y
        wx, wy = renderer.unproject_point_at_z(x, y, z_plane=-1.40)
        mouse_pos[0] = max(-0.85, min(0.85, wx))
        mouse_pos[1] = max(0.04, min(0.55, wy))
        mouse_last_moved[0] = time.time()

        if event == cv2.EVENT_LBUTTONDOWN:
            click_swing[0] = True
            last_click_pos[0] = (x, y)

    # Try opening webcam, fallback to synthetic camera
    cam = WebcamSource(device_index=0, width=640, height=480)
    if not cam.is_opened:
        print("[AVISO] Webcam física não detectada. Ativando Câmera Sintética...")
        cam = SyntheticCameraSource(width=640, height=480)

    window_name = "AI Ping Pong - Wii Sports Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse_event)

    # Center window cleanly on desktop
    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
            win_x = max(10, (screen_w - renderer.width) // 2)
            win_y = max(20, (screen_h - renderer.height) // 2 - 25)
            cv2.moveWindow(window_name, win_x, win_y)
        except Exception:
            pass

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

            # 2. Track hand via MediaPipe (gesture-cv-ux)
            track_res = tracker.process_frame(frame)

            # 2.0 Menu and HUD Mouse Click Handling
            if click_swing[0]:
                cx, cy = last_click_pos[0]
                if in_menu:
                    click_swing[0] = False
                    action = renderer.get_menu_action_at(cx, cy)
                    if action == "start":
                        if game.status == MatchStatus.READY:
                            game.prepare_serve()
                            sounds.play_cheer()
                            game.announcement = "PREPARAR... SAQUE!"
                            game.announcement_timer = 2.0
                        elif game.status == MatchStatus.GAME_OVER:
                            game.start_next_set()
                            sounds.play_bounce(ball_x=game.ball.x)
                        else:
                            game.announcement = "CONTINUANDO..."
                            game.announcement_timer = 1.0
                        in_menu = False
                        print("[MENU] Partida iniciada via clique!")
                    elif action in ("diff_1", "diff_2", "diff_3"):
                        diff_map = {"diff_1": "EASY", "diff_2": "MEDIUM", "diff_3": "HARD"}
                        d_val = diff_map[action]
                        game.start_new_match(d_val)
                        sounds.play_bounce()
                        print(f"[MENU] Dificuldade alterada para {d_val}")
                    elif action == "restart":
                        game.start_new_match(game.difficulty)
                        sounds.play_bounce()
                        print("[MENU] Placar reiniciado (0 x 0)!")
                    elif action == "calibrate":
                        in_menu = False
                        trainer.start()
                        print("[MENU] Modo de treino iniciado!")
                else:
                    if renderer.is_hud_menu_clicked(cx, cy):
                        in_menu = True
                        click_swing[0] = False
                        print("[MENU] Partida pausada via HUD [M] MENU!")

            # 2.1. Physical Racket Grip Detection & Match Flow
            current_fist = track_res.detected and (track_res.is_fist or getattr(track_res, "is_grip", False))
            if not in_menu:
                if current_fist and not prev_fist and game.status == MatchStatus.GAME_OVER:
                    game.start_next_set()
                    sounds.play_bounce(ball_x=game.ball.x)
                    print(f"[JOGO] Mão pronta! Iniciando Set {game.set_number} contra Roberval!")
            prev_fist = current_fist

            # 2.2. Update Gesture Trainer if active (gesture-cv-ux calibration mode)
            if trainer.is_active or trainer.feedback_timer > 0:
                if track_res.detected:
                    raw_spd = math.sqrt(track_res.vx**2 + track_res.vy**2 + track_res.vz**2)
                    cur_speed = raw_spd
                    cur_vx = track_res.vx
                    cur_vy = track_res.vy
                else:
                    cur_speed = 0.0
                    cur_vx = 0.0
                    cur_vy = 0.0
                cur_fist = track_res.is_fist if track_res.detected else False
                completed = trainer.update(cur_speed, cur_vx, cur_vy, cur_fist, dt)
                if getattr(trainer, "just_recorded_rep", False):
                    trainer.just_recorded_rep = False
                    sounds.play_bounce()
                if completed:
                    tracker.gesture_classifier.apply_profile(trainer.profile)
                    sounds.play_cheer()
                    print("[TREINO] Calibração de movimentos do braço concluída!")

            # Determine paddle position: Mouse takes priority if actively moved recently, else hand
            use_mouse = (now - mouse_last_moved[0]) < 1.2
            if use_mouse or not track_res.detected:
                target_x = mouse_pos[0]
                target_y = mouse_pos[1]
                target_z = -1.40
                game.player_paddle.hand_detected = False
                game.player_paddle.hand_speed = 0.0
                game.player_paddle.is_hook_swing = (game.player_paddle.speed >= 0.80)
                game.player_paddle.hook_power = max(1.0, 1.0 + game.player_paddle.speed * 0.35)
                game.player_paddle.stroke_type = None
            else:
                target_x = track_res.world_x
                target_y = track_res.world_y
                target_z = track_res.world_z
                game.player_paddle.hand_detected = True
                game.player_paddle.hand_speed = getattr(track_res, "hook_power", 0.0)
                game.player_paddle.is_hook_swing = (
                    getattr(track_res, "is_hook_swing", False)
                    or (game.player_paddle.speed >= 0.55)
                    or (track_res.speed_kmh >= 14.0)
                )
                game.player_paddle.hook_power = max(
                    getattr(track_res, "hook_power", 1.0),
                    1.0 + game.player_paddle.speed * 0.40,
                )
                if track_res.gesture and track_res.gesture.is_active_swing:
                    game.player_paddle.stroke_type = track_res.gesture.stroke_type
                else:
                    game.player_paddle.stroke_type = None

            # 1:1 Natural Hand Kinematics: The paddle smoothly and directly tracks the player's physical hand
            # Motion Attack Model in physics.py handles broad strike reach during swings without unnatural rubber-banding
            pass

            if use_mouse or not track_res.detected:
                game.player_paddle.update_position(target_x, target_y, target_z, dt)
            else:
                game.player_paddle.update_position(
                    target_x,
                    target_y,
                    target_z,
                    dt,
                    roll=track_res.roll,
                    pitch=track_res.pitch,
                    yaw=track_res.yaw,
                    normal=track_res.normal,
                    arm_power=getattr(track_res, "arm_power", 1.0),
                    elbow_angle=getattr(track_res, "elbow_angle", 0.0),
                    arm_detected=getattr(track_res, "arm_detected", False),
                    kinetic_phase=getattr(track_res, "kinetic_phase", "IDLE"),
                )

            # 2.2. SERVE BY PHYSICAL HAND/ARM SWING MOTION & FORCE
            if game.status == MatchStatus.SERVING and game.server == "PLAYER" and getattr(game, "serve_cooldown", 0.0) <= 0:
                arm_speed = game.player_paddle.speed if not track_res.detected else max(
                    getattr(track_res, "hook_power", game.player_paddle.speed),
                    game.player_paddle.speed,
                    track_res.speed_kmh / 3.6,
                )
                is_swing = (
                    getattr(track_res, "is_hook_swing", False)
                    or (arm_speed >= 0.50)
                    or (getattr(track_res, "vz", 0.0) > 0.25)
                    or (getattr(track_res, "vy", 0.0) > 0.25)
                    or (game.player_paddle.vz > 0.35)
                    or (game.player_paddle.speed >= 0.50)
                )

                dx = game.ball.x - game.player_paddle.x
                dy = game.ball.y - game.player_paddle.y
                dist_xy = math.sqrt(dx * dx + dy * dy)

                # In Motion Attack Mode, any deliberate arm swing strikes the floating serve immediately
                if is_swing or (dist_xy <= 0.65 and arm_speed >= 0.40) or (dist_xy <= 0.35):
                    vy_lift = game.player_paddle.vy if game.player_paddle.vy > 0 else 0.55
                    game.player_paddle.x = game.ball.x
                    game.player_paddle.y = game.ball.y
                    game.player_paddle.z = game.ball.z - 0.03
                    game.strike_serve(
                        swing_speed=arm_speed,
                        vx_offset=float(np.clip(track_res.world_x - game.ball.x, -0.4, 0.4)) if track_res.detected else 0.0,
                        vy_lift=vy_lift,
                    )
                    b_proj = renderer.project_point(game.ball.x, game.ball.y, game.ball.z)
                    sx, sy = b_proj[:2] if b_proj else (450, 480)
                    sounds.trigger_hit_spark(sx, sy, "SERVE", ball_x=game.ball.x)
                    renderer.add_trauma(0.35)
                    print(f"[SAQUE] Golpe executado com forca do braco: {arm_speed:.2f} m/s ({game.ball.speed_kmh:.1f} km/h)")

            # 3. Handle Mouse Click / Buffered Action (input-systems)
            if click_swing[0]:
                click_swing[0] = False
                click_buffer_timer = 0.16  # 160ms buffer window

            if click_buffer_timer > 0:
                click_buffer_timer -= dt
                if game.status == MatchStatus.READY:
                    click_buffer_timer = 0.0
                    game.prepare_serve()
                    sounds.play_bounce(ball_x=game.ball.x)
                elif game.status == MatchStatus.GAME_OVER:
                    click_buffer_timer = 0.0
                    game.start_next_set()
                    sounds.play_bounce(ball_x=game.ball.x)
                elif game.status == MatchStatus.SERVING and game.server == "PLAYER":
                    if getattr(game, "serve_cooldown", 0.0) <= 0:
                        click_buffer_timer = 0.0
                        arm_speed = max(2.4, game.player_paddle.speed)
                        game.player_paddle.x = game.ball.x
                        game.player_paddle.y = game.ball.y
                        game.player_paddle.z = game.ball.z - 0.03
                        game.strike_serve(swing_speed=arm_speed, vx_offset=(game.ball.x - game.player_paddle.x), vy_lift=0.6)
                        b_proj = renderer.project_point(game.ball.x, game.ball.y, game.ball.z)
                        sx, sy = b_proj[:2] if b_proj else (450, 480)
                        sounds.trigger_hit_spark(sx, sy, "SERVE", ball_x=game.ball.x)
                        renderer.add_trauma(0.35)
                elif game.status == MatchStatus.RALLY and game.ball.vz < 0 and game.ball.z <= -0.75:
                    # Execute buffered power smash / motion attack
                    click_buffer_timer = 0.0
                    game.player_paddle.attack_triggered = True
                    game.player_paddle.vx = float(np.clip((game.ball.x - game.player_paddle.x) * 3.5, -1.0, 1.0))
                    game.player_paddle.vy = 0.4
                    game.player_paddle.vz = 5.0
                    game.player_paddle.is_hook_swing = True
                    game.player_paddle.hook_power = 2.4

            # 4. Step game simulation with Hit-Stop (game-feel)
            if trainer.is_active or in_menu:
                # Training mode or Menu active: pause match simulation so user can configure or practice freely
                pass
            elif hit_stop_timer > 0.0:
                hit_stop_timer -= dt
                # Micro-pause on smash: skip physics step to let impact sink in
            else:
                prev_rally = game.current_rally_count
                prev_bounces = game.ball.bounce_count_current_hit
                prev_p_score = game.player_score
                prev_ai_score = game.ai_score

                game.update(dt=dt)

                if game.player_score > prev_p_score:
                    sounds.play_point_scored(winner="PLAYER")
                    renderer.add_trauma(0.25)
                elif game.ai_score > prev_ai_score:
                    sounds.play_point_scored(winner="ROBERVAL")
                    renderer.add_trauma(0.18)

                if game.current_rally_count > prev_rally:
                    shot_type = game.current_shot.shot_type if game.current_shot else "DRIVE"
                    is_smash = ("SMASH" in shot_type)
                    b_proj = renderer.project_point(game.ball.x, game.ball.y, game.ball.z)
                    sx, sy = b_proj[:2] if b_proj else (450, 320)
                    sounds.trigger_hit_spark(sx, sy, shot_type, ball_x=game.ball.x)
                    renderer.add_trauma(0.60 if is_smash else 0.22)
                    if is_smash:
                        hit_stop_timer = 0.042  # 42ms hit-stop punch
                    if game.current_rally_count in (6, 10, 15):
                        sounds.play_cheer()
                elif game.ball.bounce_count_current_hit > prev_bounces:
                    sounds.play_bounce(ball_x=game.ball.x)

            # 5. Render 3D Perspective game world with lookahead camera (camera-systems)
            renderer.update_camera(game.ball, dt)
            scene_frame = renderer.render_frame(game, track_res=track_res, in_menu=in_menu, mouse_screen_pos=(mouse_screen[0], mouse_screen[1]))
            sounds.update_and_render(scene_frame, dt)

            # 6. Composite Camera Picture-in-Picture (gesture-cv-ux)
            cw, ch = (195, 146) if in_menu else (190, 142)
            cam_view = cv2.resize(track_res.annotated_frame, (cw, ch))
            sh, sw = scene_frame.shape[:2]
            if in_menu:
                cam_x1 = 740
                cam_y1 = 35
            else:
                cam_x1 = sw - cw - 15
                cam_y1 = 82
            cam_x2 = cam_x1 + cw
            cam_y2 = cam_y1 + ch

            scene_frame[cam_y1:cam_y2, cam_x1:cam_x2] = cam_view

            # Dynamic tracking confidence border & status
            # Dynamic tracking confidence border & status
            arm_label = getattr(track_res, "playing_arm", "DIR")
            if track_res.detected:
                if getattr(track_res, "is_hook_swing", False) or track_res.speed_kmh > 30:
                    pip_border_color = (255, 230, 0)  # Neon Cyan BGR
                    pip_badge_text = "SWING ATIVO ⚡"
                elif getattr(track_res, "arm_detected", False):
                    pip_border_color = (100, 240, 100)  # Lime Green BGR
                    pip_badge_text = f"BRACO {arm_label} ({int(track_res.elbow_angle)}deg)"
                else:
                    pip_border_color = (100, 240, 100)  # Lime Green BGR
                    pip_badge_text = f"TRACKING: {arm_label}"
            elif getattr(track_res, "both_shoulders_detected", False) or getattr(track_res, "arm_detected", False):
                pip_border_color = (0, 215, 255)  # Gold BGR
                pip_badge_text = f"CORPO/OMBROS OK ({arm_label})"
            else:
                pip_border_color = (70, 70, 240)  # Coral Red BGR
                pip_badge_text = "MAO AUSENTE"

            cv2.rectangle(
                scene_frame,
                (cam_x1, cam_y1),
                (cam_x2, cam_y2),
                pip_border_color,
                2,
            )
            # Small discrete badge for camera
            cv2.rectangle(scene_frame, (cam_x1, cam_y1 - 18), (cam_x1 + 130, cam_y1), (25, 25, 30), -1)
            cv2.putText(scene_frame, pip_badge_text, (cam_x1 + 6, cam_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, pip_border_color, 1)

            # Hand Depth Strike Zone HUD Needle (gesture-cv-ux)
            bar_x = cam_x1 - 16
            bar_w = 8
            bar_y = cam_y1
            bar_h = ch
            cv2.rectangle(scene_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (20, 25, 35), -1)
            sweet_y1 = int(bar_y + bar_h * 0.25)
            sweet_y2 = int(bar_y + bar_h * 0.75)
            cv2.rectangle(scene_frame, (bar_x, sweet_y1), (bar_x + bar_w, sweet_y2), (40, 140, 40), -1)
            cv2.rectangle(scene_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 95, 115), 1)
            cv2.putText(scene_frame, "Z", (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 200, 220), 1)

            if track_res.detected:
                z_clamped = max(-1.70, min(-0.90, track_res.world_z))
                needle_ratio = (z_clamped - (-1.70)) / 0.80
                needle_y = int(bar_y + bar_h * (1.0 - needle_ratio))
                if sweet_y1 <= needle_y <= sweet_y2:
                    needle_color = (0, 255, 120)  # Optimal strike zone
                elif needle_y < sweet_y1:
                    needle_color = (255, 200, 0)  # Too deep
                else:
                    needle_color = (0, 160, 255)  # Too close
                cv2.line(scene_frame, (bar_x - 3, needle_y), (bar_x + bar_w + 3, needle_y), needle_color, 2)

            # 6.2. User Swing Speed & Power HUD Pod (gesture-cv-ux & game-feel)
            gauge_y = cam_y2 + 8
            gauge_w = cw
            gauge_h = 36
            overlay = scene_frame.copy()
            cv2.rectangle(overlay, (cam_x1, gauge_y), (cam_x1 + gauge_w, gauge_y + gauge_h), (18, 22, 32), -1)
            cv2.addWeighted(overlay, 0.85, scene_frame, 0.15, 0, scene_frame)
            cv2.rectangle(scene_frame, (cam_x1, gauge_y), (cam_x1 + gauge_w, gauge_y + gauge_h), (50, 65, 85), 1)

            user_spd = max(game.player_paddle.speed, getattr(game.player_paddle, "hand_speed", 0.0))
            user_spd_kmh = user_spd * 3.6
            if user_spd >= 2.4:
                tier_text = "SMASH!"
                tier_color = (0, 140, 255)
            elif user_spd >= 1.0:
                tier_text = "DRIVE"
                tier_color = (255, 220, 0)
            else:
                tier_text = "PRONTO"
                tier_color = (120, 240, 120)

            spd_text = f"SWING: {user_spd:.1f} m/s"
            cv2.putText(scene_frame, spd_text, (cam_x1 + 8, gauge_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 240, 255), 1)
            cv2.putText(scene_frame, tier_text, (cam_x1 + gauge_w - 68, gauge_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tier_color, 1)

            # Power fill bar with tier coloring
            power_ratio = min(1.0, user_spd / 4.0)
            bar_len = int((gauge_w - 16) * power_ratio)
            bar_color = (0, 140, 255) if user_spd >= 2.4 else ((255, 230, 0) if user_spd >= 1.0 else (100, 220, 100))
            cv2.rectangle(scene_frame, (cam_x1 + 8, gauge_y + 22), (cam_x1 + 8 + bar_len, gauge_y + 28), bar_color, -1)

            # 6.3. Interactive Gesture Training HUD Overlay (gesture-cv-ux)
            if trainer.is_active or trainer.feedback_timer > 0:
                trainer.render_overlay(scene_frame, current_speed_kmh=user_spd_kmh)

            if in_menu:
                # 6.4. Virtual Racket & Hand Sensor Card (in-menu right panel)
                r_card_y = gauge_y + gauge_h + 8
                r_card_w = cw
                r_card_h = 80
                overlay = scene_frame.copy()
                cv2.rectangle(overlay, (cam_x1, r_card_y), (cam_x1 + r_card_w, r_card_y + r_card_h), (18, 24, 34), -1)
                cv2.addWeighted(overlay, 0.88, scene_frame, 0.12, 0, scene_frame)
                cv2.rectangle(scene_frame, (cam_x1, r_card_y), (cam_x1 + r_card_w, r_card_y + r_card_h), (50, 65, 85), 1)

                cv2.putText(scene_frame, "RAQUETE VIRTUAL", (cam_x1 + 10, r_card_y + 17), cv2.FONT_HERSHEY_DUPLEX, 0.36, (0, 230, 255), 1, cv2.LINE_AA)
                is_racket_on = track_res.detected and (track_res.is_fist or getattr(track_res, "is_grip", False))
                grip_lbl = getattr(track_res, "grip_type", "CLASSICO") if track_res.detected else "NENHUMA"
                status_txt = "CONECTADA [OK]" if is_racket_on else ("EM ESPERA" if track_res.detected else "CAMERA VAZIA")
                status_col = (60, 240, 120) if is_racket_on else ((0, 215, 255) if track_res.detected else (80, 100, 255))
                cv2.putText(scene_frame, f"Status: {status_txt}", (cam_x1 + 10, r_card_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.32, status_col, 1, cv2.LINE_AA)
                cv2.putText(scene_frame, f"Grip: {grip_lbl}", (cam_x1 + 10, r_card_y + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (210, 220, 235), 1, cv2.LINE_AA)
                rub_txt = "Borracha: Vermelha (FH)" if (not track_res.detected or track_res.world_x >= -0.05) else "Borracha: Preta (BH)"
                cv2.putText(scene_frame, rub_txt, (cam_x1 + 10, r_card_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 175, 195), 1, cv2.LINE_AA)

                # 6.5. Quick Shortcuts Guide (in-menu right panel)
                k_card_y = r_card_y + r_card_h + 8
                k_card_w = cw
                k_card_h = 138
                overlay = scene_frame.copy()
                cv2.rectangle(overlay, (cam_x1, k_card_y), (cam_x1 + k_card_w, k_card_y + k_card_h), (18, 22, 32), -1)
                cv2.addWeighted(overlay, 0.88, scene_frame, 0.12, 0, scene_frame)
                cv2.rectangle(scene_frame, (cam_x1, k_card_y), (cam_x1 + k_card_w, k_card_y + k_card_h), (45, 58, 76), 1)

                cv2.putText(scene_frame, "ATALHOS RAPIDOS:", (cam_x1 + 10, k_card_y + 17), cv2.FONT_HERSHEY_DUPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
                shortcuts = [
                    ("[ESPACO]", "Iniciar Partida"),
                    ("[1 - 3]", "Dificuldade"),
                    ("[H]", "Braco (Dir/Esq)"),
                    ("[M]", "Menu / Pausa"),
                    ("[T]", "Treinar Braco"),
                    ("[R]", "Reiniciar Jogo"),
                    ("[F]", "Tela Cheia"),
                ]
                for si, (s_key, s_desc) in enumerate(shortcuts):
                    sy = k_card_y + 35 + si * 15
                    cv2.putText(scene_frame, s_key, (cam_x1 + 10, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 220, 255), 1, cv2.LINE_AA)
                    cv2.putText(scene_frame, s_desc, (cam_x1 + 75, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 200, 220), 1, cv2.LINE_AA)
            else:
                # 7. Render Discrete Controls Legend on Bottom Left during gameplay
                diff_map = {"EASY": "FACIL [1]", "MEDIUM": "MEDIO [2]", "HARD": "DIFICIL [3]", "CHAMPION": "LENDA [4]"}
                cur_diff = diff_map.get(game.difficulty, "MEDIO [2]")
                action_desc = "SACAR/GOLPE"
                control_hint = f"DIFICULDADE: {cur_diff} | MODO: ATAQUE POR MOVIMENTO [Wii Sports] | BRACO: {tracker.handedness_mode} [H] | {action_desc}: [GOLPEIE/ESPACO] | MENU: [M] | TREINO: [T] | REINICIAR: [R] | TELA CHEIA: [F]"
                cv2.putText(
                    scene_frame,
                    control_hint,
                    (20, sh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (210, 215, 225),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, scene_frame)

            # 8. Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                if trainer.is_active or trainer.feedback_timer > 0:
                    trainer.stop()
                    print("[TREINO] Modo de treino fechado via ESC.")
                elif in_menu and (game.player_score > 0 or game.ai_score > 0) and key == 27:
                    # ESC while paused returns to match
                    in_menu = False
                else:
                    break
            elif key in (ord("m"), ord("M")):
                in_menu = not in_menu
                if not in_menu and game.status == MatchStatus.READY:
                    game.prepare_serve()
                    sounds.play_cheer()
                    game.announcement = "PREPARAR... SAQUE!"
                    game.announcement_timer = 2.0
                print(f"[MENU] Menu {'aberto' if in_menu else 'fechado'}")
            elif key in (ord("h"), ord("H")):
                new_mode = tracker.toggle_handedness()
                mode_names = {"AUTO": "AUTO (AUTOMATICO)", "RIGHT": "DESTRO (DIREITA)", "LEFT": "CANHOTO (ESQUERDA)"}
                lbl = mode_names.get(new_mode, new_mode)
                game.announcement = f"BRACO: {lbl}"
                game.announcement_timer = 2.0
                print(f"[CONTROLE] Modo de braco alterado para: {lbl}")
            elif key in (ord("t"), ord("T")):
                if trainer.is_active or trainer.feedback_timer > 0:
                    trainer.stop()
                    print("[TREINO] Modo de treino encerrado.")
                else:
                    in_menu = False
                    trainer.start()
                    print("[TREINO] Modo de treino iniciado! Siga as instrucoes na tela.")
            elif key in (ord("z"), ord("Z")):
                tracker.auto_depth = not tracker.auto_depth
                state_str = "ATIVADA (3D)" if tracker.auto_depth else "FIXA (-1.40m)"
                game.announcement = f"PROFUNDIDADE Z: {state_str}"
                game.announcement_timer = 2.0
                print(f"[CONTROLE] Profundidade Z {state_str}")
            elif key in (ord("f"), ord("F")):
                is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if is_fullscreen else cv2.WINDOW_FULLSCREEN,
                )
            elif key in (32, 10, 13):  # Space or Enter
                if trainer.is_active:
                    trainer.skip_step()
                elif in_menu:
                    if game.status == MatchStatus.READY:
                        game.prepare_serve()
                        sounds.play_cheer()
                        game.announcement = "PREPARAR... SAQUE!"
                        game.announcement_timer = 2.0
                    elif game.status == MatchStatus.GAME_OVER:
                        game.start_next_set()
                        sounds.play_bounce(ball_x=game.ball.x)
                    else:
                        game.announcement = "CONTINUANDO..."
                        game.announcement_timer = 1.0
                    in_menu = False
                    print("[MENU] Partida iniciada/retomada via teclado!")
                elif game.status == MatchStatus.GAME_OVER:
                    game.start_next_set()
                    sounds.play_bounce(ball_x=game.ball.x)
                elif game.status == MatchStatus.SERVING and game.server == "PLAYER" and getattr(game, "serve_cooldown", 0.0) <= 0:
                    game.strike_serve(swing_speed=2.5, vx_offset=0.0, vy_lift=0.6)
                    b_proj = renderer.project_point(game.ball.x, game.ball.y, game.ball.z)
                    sx, sy = b_proj[:2] if b_proj else (450, 480)
                    sounds.trigger_hit_spark(sx, sy, "SERVE", ball_x=game.ball.x)
                    renderer.add_trauma(0.35)
                elif game.status == MatchStatus.RALLY:
                    # Spacebar triggers attack swing during rally
                    click_swing[0] = True
            elif key in (ord("r"), ord("R")):
                if game.stats.shots:
                    storage.save_match(game.stats)
                game.start_new_match()
                sounds.play_bounce()
                print("[JOGO] Nova partida iniciada!")
            elif key in (ord("s"), ord("S")):
                storage.save_match(game.stats)
                print(f"[DATABRICKS] Partida {game.match_id} salva.")
            elif key == ord("1"):
                game.start_new_match("EASY")
                sounds.play_bounce()
                print("[DIFICULDADE] Alterada para EASY")
            elif key == ord("2"):
                game.start_new_match("MEDIUM")
                sounds.play_bounce()
                print("[DIFICULDADE] Alterada para MEDIUM")
            elif key == ord("3"):
                game.start_new_match("HARD")
                sounds.play_bounce()
                print("[DIFICULDADE] Alterada para HARD")
            elif key == ord("4"):
                game.start_new_match("CHAMPION")
                sounds.play_bounce()
                print("[DIFICULDADE] Alterada para CHAMPION")

    finally:
        if game.stats.shots:
            storage.save_match(game.stats)
            print(f"[TELEMETRIA] Partida {game.match_id} registrada.")

        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_desktop_app()
