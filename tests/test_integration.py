"""End-to-end integration test verifying full gameplay loop, rendering and telemetry."""

import numpy as np
import pytest

from src.engine.game_state import GameStateMachine, MatchStatus
from src.ui.renderer_3d import PerspectiveRenderer3D
from src.databricks.storage import LocalDeltaStorage
from src.databricks.analytics import PingPongAnalytics


def test_full_match_simulation_and_rendering(tmp_path):
    # 1. Initialize match
    game = GameStateMachine(difficulty="EASY")
    renderer = PerspectiveRenderer3D()
    db_file = tmp_path / "integration_matches.db"
    storage = LocalDeltaStorage(db_path=str(db_file))

    assert game.status == MatchStatus.READY

    # 2. Trigger Serve
    game.trigger_serve()
    assert game.status == MatchStatus.RALLY
    assert game.ball.in_play is True
    assert len(game.stats.shots) == 1

    # 3. Simulate 60 frames of play
    for _ in range(60):
        # AI and physics step
        game.update(dt=0.016)

        # Render 3D scene
        frame = renderer.render_frame(game)
        assert frame is not None
        assert frame.shape == (renderer.height, renderer.width, 3)

    # 4. Award a point to test state transition
    game._award_point("PLAYER", "WINNER POINT!")
    assert game.player_score == 1
    # After awarding point, game resets for serve
    assert game.status == MatchStatus.SERVING

    # 5. Fast-forward to 10-9, then award winning point for 11-9 game over
    game.player_score = 10
    game.ai_score = 9
    game._award_point("PLAYER", "CHAMPIONSHIP WINNER")
    assert game.status == MatchStatus.GAME_OVER
    assert game.stats.winner == "PLAYER"
    assert game.player_score == 11

    # 6. Save match to storage
    storage.save_match(game.stats)

    # 7. Validate telemetry persistence and analytics
    saved_matches = storage.get_matches()
    assert len(saved_matches) == 1
    assert saved_matches[0]["match_id"] == game.match_id
    assert saved_matches[0]["winner"] == "PLAYER"
    assert saved_matches[0]["player_score"] == 11

    shots = storage.get_shots_by_match(game.match_id)
    breakdown = PingPongAnalytics.compute_match_breakdown(shots)
    assert breakdown["total_shots"] >= 1


def test_desktop_main_loop_simulation():
    """Simulate main.py desktop runner step-by-step to catch any NameError or runtime exceptions."""
    import math
    import cv2
    from src.vision.tracker import HandTrackingResult
    from src.vision.gestures import GestureResult

    game = GameStateMachine(difficulty="MEDIUM")
    renderer = PerspectiveRenderer3D()

    # Step 1: In READY state, simulate fist gesture to start/prepare serve
    assert game.status == MatchStatus.READY
    game.prepare_serve()
    assert game.status == MatchStatus.SERVING

    # Simulate game.update() while in SERVING state (ball floating waiting for arm stroke)
    for _ in range(5):
        game.update(dt=0.016)
    assert game.status == MatchStatus.SERVING

    # Step 2: In SERVING state, simulate HandTrackingResult with arm swing
    gesture = GestureResult(
        stroke_type="GANCHO_FOREHAND",
        swing_speed=2.8,
        swing_speed_kmh=10.0,
        is_active_swing=True,
        is_hook_swing=True,
        hook_power=2.8,
        confidence=0.9,
    )
    track_res = HandTrackingResult(
        detected=True,
        world_x=0.1,
        world_y=0.25,
        world_z=-1.40,
        vx=1.2,
        vy=0.4,
        vz=2.0,
        speed_kmh=10.0,
        gesture=gesture,
        annotated_frame=np.zeros((240, 320, 3), dtype=np.uint8),
    )

    # Execute main.py distance and serve calculation
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

    assert game.status == MatchStatus.RALLY

    # Step 3: Render frame and composite camera picture-in-picture
    scene_frame = renderer.render_frame(game)
    assert scene_frame is not None

    cw, ch = 190, 142
    cam_view = cv2.resize(track_res.annotated_frame, (cw, ch))
    sh, sw = scene_frame.shape[:2]
    cam_x1 = sw - cw - 15
    cam_y1 = 82
    cam_x2 = sw - 15
    cam_y2 = cam_y1 + ch
    scene_frame[cam_y1:cam_y2, cam_x1:cam_x2] = cam_view
    assert scene_frame.shape == (renderer.height, renderer.width, 3)


def test_pre_game_menu_modal_and_interactions():
    """Verify pre-game Start Menu / Lobby rendering, hitboxes, and action handling."""
    game = GameStateMachine(difficulty="MEDIUM")
    renderer = PerspectiveRenderer3D()

    # 1. Render in menu mode without hand tracking (empty camera)
    frame_menu = renderer.render_frame(game, track_res=None, in_menu=True)
    assert frame_menu is not None
    assert frame_menu.shape == (renderer.height, renderer.width, 3)

    # 2. Check that all expected interactive button hitboxes are registered (3 difficulty levels: EASY, MEDIUM, HARD)
    expected_buttons = ["start", "diff_1", "diff_2", "diff_3", "restart", "calibrate"]
    for btn_name in expected_buttons:
        assert btn_name in renderer.menu_buttons, f"Button {btn_name} should be in menu_buttons"
        bx1, by1, bx2, by2 = renderer.menu_buttons[btn_name]
        assert bx2 > bx1 and by2 > by1, f"Button {btn_name} has invalid geometry: ({bx1}, {by1}, {bx2}, {by2})"

    # 3. Test clicking on the start button
    sx1, sy1, sx2, sy2 = renderer.menu_buttons["start"]
    click_x = int((sx1 + sx2) / 2)
    click_y = int((sy1 + sy2) / 2)
    assert renderer.get_menu_action_at(click_x, click_y) == "start"

    # 4. Test clicking on difficulty 3 (HARD)
    dx1, dy1, dx2, dy2 = renderer.menu_buttons["diff_3"]
    click_dx = int((dx1 + dx2) / 2)
    click_dy = int((dy1 + dy2) / 2)
    assert renderer.get_menu_action_at(click_dx, click_dy) == "diff_3"

    # 5. Test clicking outside any button
    assert renderer.get_menu_action_at(5, 5) is None

    # 6. Test in-game HUD Menu button click
    hx1, hy1, hx2, hy2 = renderer.hud_menu_button
    hud_cx = int((hx1 + hx2) / 2)
    hud_cy = int((hy1 + hy2) / 2)
    assert renderer.is_hud_menu_clicked(hud_cx, hud_cy) is True
    assert renderer.is_hud_menu_clicked(900, 500) is False
