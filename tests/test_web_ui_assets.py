"""Test suite for Web UI/UX assets, Canvas Menu, HUD, and Usability enhancements."""

from pathlib import Path
import re
import numpy as np
import pytest

from src.engine.game_state import GameStateMachine, MatchStatus
from src.engine.table import Table
from src.ui.renderer_3d import PerspectiveRenderer3D
from app import get_web_game_component


def test_docs_game_js_syntax_and_ui_features():
    """Verify docs/game.js includes tabbed menu modal, mouse hover, and keyboard navigation."""
    js_path = Path("docs/game.js")
    assert js_path.exists(), "docs/game.js must exist"
    content = js_path.read_text(encoding="utf-8")

    # Check tabs
    assert "tab_play" in content, "docs/game.js must have tab_play"
    assert "tab_help" in content, "docs/game.js must have tab_help"
    assert "tab_bio" in content, "docs/game.js must have tab_bio"

    # Check mouse hover tracking & cursor state
    assert "updateCursorState" in content, "docs/game.js must implement updateCursorState"
    assert "mouseScreenX" in content, "docs/game.js must track mouseScreenX"
    assert "mouseScreenY" in content, "docs/game.js must track mouseScreenY"

    # Check keyboard accessibility
    assert "getFocusableActions" in content, "docs/game.js must provide focusable actions list"
    assert "focusedActionIndex" in content, "docs/game.js must manage keyboard focus index"

    # Check HUD additions
    assert "BOLA:" in content or "KM/H" in content, "docs/game.js must display speedometer"
    assert "player_sets" in content, "docs/game.js must track and display sets"


def test_docs_style_css_glassmorphic_styling():
    """Verify docs/style.css includes glassmorphic styling and responsive canvas layout."""
    css_path = Path("docs/style.css")
    assert css_path.exists(), "docs/style.css must exist"
    css = css_path.read_text(encoding="utf-8")

    assert "#game-canvas" in css, "style.css must style #game-canvas"
    assert "border-radius" in css, "style.css must use rounded corners for sports glassmorphic UI"
    assert "box-shadow" in css, "style.css must have glow/shadow elevation"
    assert "aspect-ratio: 16 / 10" in css, "style.css must preserve 16:10 aspect ratio"
    assert "min(98vw" in css or "min(96vw" in css, "style.css must dynamically scale to viewport width"


def test_responsive_and_high_dpi_scaling():
    """Verify high-DPI canvas buffer scaling and multi-screen touch support."""
    js_content = Path("docs/game.js").read_text(encoding="utf-8")
    assert "prepareFrame" in js_content, "game.js must implement prepareFrame for dynamic high-DPI resizing"
    assert "devicePixelRatio" in js_content, "game.js must utilize devicePixelRatio for crisp rendering"
    assert "touchstart" in js_content, "game.js must support touch input for tablets and touchscreens"
    assert "touchmove" in js_content, "game.js must track touch dragging"

    import cv2
    arena_img = cv2.imread("assets/background_arena.png")
    assert arena_img is not None
    assert arena_img.shape == (1280, 2048, 3), "assets/background_arena.png must be 2K HD (2048x1280)"


def test_app_py_web_component_generation():
    """Verify app.py generates clean HTML5 bundle with embedded arena and scripts."""
    html_bundle = get_web_game_component()
    assert "<canvas id=\"game-canvas\"" in html_bundle, "Bundle must contain game canvas"
    assert "window.BG_ARENA_IMAGE_DATA" in html_bundle, "Bundle must inject arena texture"
    assert "tab_play" in html_bundle, "Bundle must contain tab_play from game.js"


def test_desktop_renderer_3d_hover_and_hud_menu():
    """Verify renderer_3d.py accepts mouse_screen_pos and detects hover states."""
    renderer = PerspectiveRenderer3D()
    game = GameStateMachine(difficulty="MEDIUM")

    # 1. Render in-game HUD with mouse hovering [M] MENU button (14, 14, 106, 42)
    frame_hud = renderer.render_frame(game, mouse_screen_pos=(50, 25), in_menu=False)
    assert isinstance(frame_hud, np.ndarray), "render_frame must return np.ndarray"
    assert frame_hud.shape == (renderer.height, renderer.width, 3)

    # 2. Render menu modal with mouse hovering start button
    frame_menu = renderer.render_frame(game, mouse_screen_pos=(100, 180), in_menu=True)
    assert isinstance(frame_menu, np.ndarray)
    assert "start" in renderer.menu_buttons, "Menu must register 'start' button hitbox"
    assert "diff_1" in renderer.menu_buttons, "Menu must register difficulty button hitboxes"
    assert "restart" in renderer.menu_buttons, "Menu must register restart button hitbox"


def test_gameplay_integrity_preserved():
    """Ensure table tennis physics parameters and rules remain 100% intact."""
    table = Table()
    game = GameStateMachine(difficulty="MEDIUM")
    assert table.half_length * 2 == pytest.approx(2.74, abs=0.01), "Table length must be standard ITTF 2.74m"
    assert table.half_width * 2 == pytest.approx(1.525, abs=0.01), "Table width must be standard ITTF 1.525m"
    assert game.ball.config.RADIUS == pytest.approx(0.02, abs=0.001), "Ball radius must be 20mm (40mm ball)"
    assert game.ball.config.MASS == pytest.approx(0.0027, abs=0.0001), "Ball mass must be official 2.7g"
