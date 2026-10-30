"""3D Perspective Projection Renderer with Wii Sports table tennis aesthetics."""

import math
import os
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.config import DisplayConfig, TableConfig
from src.engine.physics import Ball
from src.engine.paddle import Paddle
from src.engine.game_state import GameStateMachine, MatchStatus


class PerspectiveRenderer3D:
    """Renders 3D table tennis game world to 2D image buffer."""

    def __init__(
        self,
        display_config: Optional[DisplayConfig] = None,
        table_config: Optional[TableConfig] = None,
    ):
        self.config = display_config or DisplayConfig()
        self.table_config = table_config or TableConfig()

        self.width = self.config.CANVAS_WIDTH
        self.height = self.config.CANVAS_HEIGHT

        # Check for realistic arena background image
        bg_candidate_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "assets", "background_arena.png"),
            "assets/background_arena.png",
        ]
        loaded_bg = None
        for p in bg_candidate_paths:
            if os.path.exists(p):
                loaded_bg = cv2.imread(p)
                if loaded_bg is not None:
                    break

        if loaded_bg is not None:
            self.has_image_background = True
            if loaded_bg.shape[1] != self.width or loaded_bg.shape[0] != self.height:
                loaded_bg = cv2.resize(loaded_bg, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.cached_background = loaded_bg

            # Exact sub-pixel calibrated 3D camera projection parameters for background_arena.png
            self.cam_base_x = 0.0
            self.cam_base_y = 1.23505274
            self.cam_base_z = -3.86738472
            self.cam_x = 0.0
            self.cam_y = 1.23505274
            self.cam_z = -3.86738472
            self.cam_pitch = math.radians(-14.9153403)
            self.focal_length = 1269.75295
            self.fov = 2.0 * math.atan((self.width / 2.0) / self.focal_length)
        else:
            self.has_image_background = False
            # Camera placement: Procedural 3D perspective looking down at table
            self.cam_base_x = 0.0
            self.cam_base_y = 1.25
            self.cam_base_z = -3.50
            self.cam_x = 0.0
            self.cam_y = 1.25
            self.cam_z = -3.50
            self.cam_pitch = math.radians(-18.0)  # Looking down 18 degrees
            self.fov = math.radians(48.0)
            self.focal_length = (self.width / 2.0) / math.tan(self.fov / 2.0)
            self.cached_background = self._create_cached_background()

        # Game Feel (Juice): Decaying screen shake trauma
        self.trauma: float = 0.0
        self.shake_time: float = 0.0

        # Menu interaction hitboxes (game-ui-ux)
        self.menu_buttons = {}  # Dict of button_name -> (x1, y1, x2, y2)
        self.hud_menu_button = (14, 14, 106, 42)  # [ ⏸ MENU ] in-game HUD button

    def _create_cached_background(self) -> np.ndarray:
        """Pre-compute tournament stadium arena background with overhead lighting atmosphere."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Deep championship arena gradient: dark ceiling to stadium tier
        top_color = np.array([24, 18, 16], dtype=np.float32)       # Dark stadium ceiling BGR
        bot_color = np.array([45, 38, 42], dtype=np.float32)       # Lower spectator stands BGR
        ratios = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None, None]
        grad = (1.0 - ratios) * top_color + ratios * bot_color
        frame[:] = grad.astype(np.uint8)

        # Distant stadium architecture: subtle spectator seating tiers
        tier_y = int(self.height * 0.32)
        cv2.line(frame, (0, tier_y), (self.width, tier_y), (35, 30, 32), 1)
        cv2.line(frame, (0, tier_y + 18), (self.width, tier_y + 18), (40, 34, 38), 1)
        cv2.line(frame, (0, tier_y + 36), (self.width, tier_y + 36), (46, 40, 44), 1)

        # Overhead spotlight glow centered above table (Wii Sports & Table Tennis Touch feel)
        center_x = self.width // 2
        center_y = int(self.height * 0.40)
        spot_overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.ellipse(spot_overlay, (center_x, center_y), (int(self.width * 0.46), int(self.height * 0.32)), 0, 0, 360, (65, 55, 50), -1)
        cv2.ellipse(spot_overlay, (center_x, center_y), (int(self.width * 0.28), int(self.height * 0.20)), 0, 0, 360, (90, 80, 75), -1)
        cv2.GaussianBlur(spot_overlay, (91, 91), 0, dst=spot_overlay)
        cv2.add(frame, spot_overlay, dst=frame)

        return frame

    def add_trauma(self, amount: float):
        """Add camera shake trauma (0.0 to 1.0) per game-feel skill."""
        self.trauma = min(1.0, self.trauma + amount)

    def update_camera(self, ball: Ball, dt: float):
        """Smooth lookahead follow and trauma decay (camera-systems & game-feel)."""
        self.shake_time += dt
        if self.trauma > 0.0:
            self.trauma = max(0.0, self.trauma - dt * 2.2)

        if not self.has_image_background:
            # Subtle lookahead follow on ball X with gentle deadzone
            if ball.in_play:
                target_x = float(np.clip(ball.x * 0.16, -0.22, 0.22))
                target_y = float(1.25 + np.clip(ball.y * 0.08, -0.05, 0.15))
            else:
                target_x = 0.0
                target_y = 1.25

            alpha = min(1.0, dt * 3.5)
            self.cam_x += (target_x - self.cam_x) * alpha
            self.cam_y += (target_y - self.cam_y) * alpha
        else:
            # Keep camera solidly locked to background table geometry
            self.cam_x = self.cam_base_x
            self.cam_y = self.cam_base_y
            self.cam_z = self.cam_base_z

    def update_shake(self, dt: float):
        """Backward compatible shake updater."""
        self.shake_time += dt
        if self.trauma > 0.0:
            self.trauma = max(0.0, self.trauma - dt * 2.2)

    def project_point(self, x: float, y: float, z: float) -> Optional[Tuple[int, int, float]]:
        """Project 3D world coordinate (meters) to 2D screen coordinate (pixels)."""
        # Relative to camera
        dx = x - self.cam_x
        dy = y - self.cam_y
        dz = z - self.cam_z

        # Pitch rotation
        cos_p = math.cos(self.cam_pitch)
        sin_p = math.sin(self.cam_pitch)

        cam_y = dy * cos_p - dz * sin_p
        cam_z = dy * sin_p + dz * cos_p

        # Behind camera clip
        if cam_z <= 0.15:
            return None

        # Perspective projection
        screen_x = int(self.width / 2.0 + (self.focal_length * dx) / cam_z)
        screen_y = int(self.height / 2.0 - (self.focal_length * cam_y) / cam_z)

        # Quadratic screen shake offset (used in procedural mode; image background uses frame warp)
        if not self.has_image_background and self.trauma > 0.0:
            shake = self.trauma * self.trauma
            ox = int(shake * 14.0 * math.sin(self.shake_time * 26.0))
            oy = int(shake * 9.0 * math.cos(self.shake_time * 32.0))
            screen_x += ox
            screen_y += oy

        return screen_x, screen_y, cam_z

    def unproject_point_at_z(self, screen_x: float, screen_y: float, z_plane: float = -1.40) -> Tuple[float, float]:
        """Unproject 2D screen coordinate to exact 3D world (X, Y) on plane Z."""
        dz = z_plane - self.cam_z
        cos_p = math.cos(self.cam_pitch)
        sin_p = math.sin(self.cam_pitch)

        vy = (self.height / 2.0 - screen_y) / self.focal_length
        denom = cos_p - vy * sin_p
        if abs(denom) < 1e-6:
            return 0.0, 0.20

        dy = dz * (vy * cos_p + sin_p) / denom
        world_y = self.cam_y + dy
        cam_z = dy * sin_p + dz * cos_p

        vx = (screen_x - self.width / 2.0) / self.focal_length
        world_x = self.cam_x + vx * cam_z
        return float(world_x), float(world_y)

    def render_frame(
        self,
        game_state: GameStateMachine,
        track_res: Optional[object] = None,
        in_menu: bool = False,
        mouse_screen_pos: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Render complete game scene to BGR image buffer."""
        # 1. Background (Pre-cached tournament arena atmosphere or photorealistic image)
        frame = self._render_background()

        # Render procedural environment only if no image background is active
        if not self.has_image_background:
            # 2. Olympic Taraflex Court Floor
            self._render_floor(frame)

            # 2.1. Court Surrounds / Barriers
            self._render_surrounds(frame)

            # 3. 3D Table Structure (Undercarriage, Legs, Apron & Top Surface)
            self._render_table(frame)

            # 4. 3D Net with Steel Posts and Clamps
            self._render_net(frame)

        # 4.1. Landing Target & Anticipation Reticle
        self._render_landing_target(frame, game_state.ball)

        # 5. Shadows
        self._render_shadows(frame, game_state.ball)

        # 6. Motion Trails
        self._render_ball_trail(frame, game_state.ball)

        # 7. AI Paddle
        self._render_paddle(frame, game_state.ai_paddle, is_player=False)

        # 8. 3D Ball with squash & stretch
        self._render_ball(frame, game_state.ball)

        # 9. Player Paddle (Rendered in front with dynamic power aura)
        self._render_paddle(frame, game_state.player_paddle, is_player=True)

        # 9.1. Strike Guide & Hitting Assist Prompt
        self._render_strike_guide(frame, game_state)

        # 10. Floating Arcade Pop-in Announcement
        self._render_announcement_pop(frame, game_state)

        # 11. Wii Sports Broadcast HUD
        if not in_menu:
            self._render_hud(frame, game_state, mouse_screen_pos=mouse_screen_pos)

        # 12. Menu Modal Overlay for Start Menu / Lobby, Set End, or Paused Menu
        if in_menu or game_state.status in (MatchStatus.READY, MatchStatus.GAME_OVER):
            self._render_menu_modal(frame, game_state, track_res=track_res, in_menu=in_menu, mouse_screen_pos=mouse_screen_pos)

        # 13. Screen Shake Trauma (Whole frame warp for unified visual impact)
        if self.has_image_background and self.trauma > 0.0:
            shake = self.trauma * self.trauma
            ox = float(shake * 12.0 * math.sin(self.shake_time * 28.0))
            oy = float(shake * 8.0 * math.cos(self.shake_time * 34.0))
            M = np.float32([[1, 0, ox], [0, 1, oy]])
            frame = cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)

        return frame

    def _render_background(self) -> np.ndarray:
        """Return pre-cached gradient background."""
        return self.cached_background.copy()

    def _render_floor(self, frame: np.ndarray):
        """Render Olympic Taraflex sports court with official tournament boundaries."""
        # Main tournament court floor (Taraflex Burgundy/Blue sports vinyl)
        p1 = self.project_point(-6.0, -0.76, -3.8)
        p2 = self.project_point(6.0, -0.76, -3.8)
        p3 = self.project_point(6.0, -0.76, 5.0)
        p4 = self.project_point(-6.0, -0.76, 5.0)

        if p1 and p2 and p3 and p4:
            pts = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]], np.int32)
            cv2.fillPoly(frame, [pts], (48, 38, 92))  # Olympic tournament burgundy vinyl (BGR)

        # Official Play Area Inner Mat (7m x 14m Taraflex mat)
        m1 = self.project_point(-3.5, -0.76, -3.4)
        m2 = self.project_point(3.5, -0.76, -3.4)
        m3 = self.project_point(3.5, -0.76, 4.0)
        m4 = self.project_point(-3.5, -0.76, 4.0)

        if m1 and m2 and m3 and m4:
            m_pts = np.array([[m1[0], m1[1]], [m2[0], m2[1]], [m3[0], m3[1]], [m4[0], m4[1]]], np.int32)
            cv2.fillPoly(frame, [m_pts], (62, 48, 115))  # Richer court center mat
            cv2.polylines(frame, [m_pts], True, (180, 185, 205), 2, cv2.LINE_AA)

        # Soft Table Ground Shadow directly on floor
        s1 = self.project_point(-self.table_config.half_width - 0.12, -0.76, -self.table_config.half_length - 0.12)
        s2 = self.project_point(self.table_config.half_width + 0.12, -0.76, -self.table_config.half_length - 0.12)
        s3 = self.project_point(self.table_config.half_width + 0.12, -0.76, self.table_config.half_length + 0.12)
        s4 = self.project_point(-self.table_config.half_width - 0.12, -0.76, self.table_config.half_length + 0.12)
        if s1 and s2 and s3 and s4:
            s_pts = np.array([[s1[0], s1[1]], [s2[0], s2[1]], [s3[0], s3[1]], [s4[0], s4[1]]], np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [s_pts], (20, 15, 30))
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    def _render_surrounds(self, frame: np.ndarray):
        """Render 3D tournament court surround barriers (A-boards with decals)."""
        bz = 4.2
        by_b = -0.76
        by_t = -0.06
        bx_l = -3.8
        bx_r = 3.8

        b_bl = self.project_point(bx_l, by_b, bz)
        b_br = self.project_point(bx_r, by_b, bz)
        b_tr = self.project_point(bx_r, by_t, bz)
        b_tl = self.project_point(bx_l, by_t, bz)

        if b_bl and b_br and b_tr and b_tl:
            pts = np.array([[b_bl[0], b_bl[1]], [b_br[0], b_br[1]], [b_tr[0], b_tr[1]], [b_tl[0], b_tl[1]]], np.int32)
            cv2.fillPoly(frame, [pts], (42, 38, 48))
            cv2.polylines(frame, [pts], True, (130, 130, 155), 2)
            mid_y = (b_tl[1] + b_bl[1]) // 2
            cv2.line(frame, (b_tl[0] + 30, mid_y), (b_tr[0] - 30, mid_y), (100, 160, 220), 2)

    def _render_table(self, frame: np.ndarray):
        """Render 3D ITTF table: steel legs, undercarriage chassis, apron skirt, top surface and lines."""
        w2 = self.table_config.half_width
        l2 = self.table_config.half_length
        y = self.table_config.SURFACE_Y
        y_floor = -0.76
        apron_depth = 0.05  # 50mm official table apron depth

        # 1. Four Heavy-Duty Steel Legs (connecting table underside to floor)
        leg_inset_x = 0.12
        leg_inset_z = 0.18
        leg_positions = [
            (-w2 + leg_inset_x, l2 - leg_inset_z),   # NW
            (w2 - leg_inset_x, l2 - leg_inset_z),    # NE
            (w2 - leg_inset_x, -l2 + leg_inset_z),   # SE
            (-w2 + leg_inset_x, -l2 + leg_inset_z),  # SW
        ]

        steel_color = (40, 42, 48)  # Matte anthracite steel
        steel_highlight = (80, 85, 95)

        for lx, lz in leg_positions:
            top_pt = self.project_point(lx, -apron_depth, lz)
            bot_pt = self.project_point(lx, y_floor, lz)
            if top_pt and bot_pt:
                leg_w = max(4, int(18.0 / bot_pt[2]))
                cv2.line(frame, (top_pt[0], top_pt[1]), (bot_pt[0], bot_pt[1]), steel_color, leg_w)
                cv2.line(frame, (top_pt[0] - 1, top_pt[1]), (bot_pt[0] - 1, bot_pt[1]), steel_highlight, 1)

        # Crossbeams connecting North legs and South legs
        lnw_bot = self.project_point(-w2 + leg_inset_x, y_floor + 0.22, l2 - leg_inset_z)
        lne_bot = self.project_point(w2 - leg_inset_x, y_floor + 0.22, l2 - leg_inset_z)
        if lnw_bot and lne_bot:
            cv2.line(frame, (lnw_bot[0], lnw_bot[1]), (lne_bot[0], lne_bot[1]), steel_color, 3)

        lsw_bot = self.project_point(-w2 + leg_inset_x, y_floor + 0.22, -l2 + leg_inset_z)
        lse_bot = self.project_point(w2 - leg_inset_x, y_floor + 0.22, -l2 + leg_inset_z)
        if lsw_bot and lse_bot:
            cv2.line(frame, (lsw_bot[0], lsw_bot[1]), (lse_bot[0], lse_bot[1]), steel_color, 3)

        # Center undercarriage wheel chassis under the net (z = 0.0)
        c_left = self.project_point(-0.25, y_floor, 0.0)
        c_right = self.project_point(0.25, y_floor, 0.0)
        c_top = self.project_point(0.0, -apron_depth, 0.0)
        if c_left and c_right and c_top:
            cv2.line(frame, (c_left[0], c_left[1]), (c_top[0], c_top[1]), steel_color, 4)
            cv2.line(frame, (c_right[0], c_right[1]), (c_top[0], c_top[1]), steel_color, 4)
            cv2.circle(frame, (c_left[0], c_left[1]), max(2, int(6.0 / c_left[2])), (20, 20, 20), -1)
            cv2.circle(frame, (c_right[0], c_right[1]), max(2, int(6.0 / c_right[2])), (20, 20, 20), -1)

        # 2. Table Apron Skirt (50mm vertical border below surface)
        c_nw = self.project_point(-w2, y, l2)
        c_ne = self.project_point(w2, y, l2)
        c_se = self.project_point(w2, y, -l2)
        c_sw = self.project_point(-w2, y, -l2)

        a_nw = self.project_point(-w2, -apron_depth, l2)
        a_ne = self.project_point(w2, -apron_depth, l2)
        a_se = self.project_point(w2, -apron_depth, -l2)
        a_sw = self.project_point(-w2, -apron_depth, -l2)

        apron_color = (115, 52, 10)  # Dark ITTF navy apron BGR
        apron_edge = (190, 190, 210)

        # Front apron (facing player, z = -l2)
        if c_sw and c_se and a_se and a_sw:
            front_pts = np.array([[c_sw[0], c_sw[1]], [c_se[0], c_se[1]], [a_se[0], a_se[1]], [a_sw[0], a_sw[1]]], np.int32)
            cv2.fillPoly(frame, [front_pts], apron_color)
            cv2.polylines(frame, [front_pts], True, apron_edge, 1)

        # Left apron
        if c_sw and c_nw and a_nw and a_sw:
            left_pts = np.array([[c_sw[0], c_sw[1]], [c_nw[0], c_nw[1]], [a_nw[0], a_nw[1]], [a_sw[0], a_sw[1]]], np.int32)
            cv2.fillPoly(frame, [left_pts], (int(apron_color[0] * 0.85), int(apron_color[1] * 0.85), int(apron_color[2] * 0.85)))
            cv2.polylines(frame, [left_pts], True, apron_edge, 1)

        # Right apron
        if c_se and c_ne and a_ne and a_se:
            right_pts = np.array([[c_se[0], c_se[1]], [c_ne[0], c_ne[1]], [a_ne[0], a_ne[1]], [a_se[0], a_se[1]]], np.int32)
            cv2.fillPoly(frame, [right_pts], (int(apron_color[0] * 0.85), int(apron_color[1] * 0.85), int(apron_color[2] * 0.85)))
            cv2.polylines(frame, [right_pts], True, apron_edge, 1)

        # 3. Top Playing Surface & Official ITTF Markings
        if c_nw and c_ne and c_se and c_sw:
            pts = np.array([[c_nw[0], c_nw[1]], [c_ne[0], c_ne[1]], [c_se[0], c_se[1]], [c_sw[0], c_sw[1]]], np.int32)
            # Championship ITTF Blue
            cv2.fillPoly(frame, [pts], (165, 82, 14))

            # Subtle specular table sheen from stadium lighting (Wii Sports polish)
            light_poly = frame.copy()
            cv2.fillPoly(light_poly, [pts], (195, 115, 35))
            cv2.addWeighted(light_poly, 0.22, frame, 0.78, 0, frame)

            # White outer boundary lines (20mm ITTF standard)
            cv2.polylines(frame, [pts], True, (255, 255, 255), 3, cv2.LINE_AA)

            # Center dividing line (3mm ITTF standard)
            c_mid_n = self.project_point(0, y, l2)
            c_mid_s = self.project_point(0, y, -l2)
            if c_mid_n and c_mid_s:
                cv2.line(frame, (c_mid_n[0], c_mid_n[1]), (c_mid_s[0], c_mid_s[1]), (255, 255, 255), 2, cv2.LINE_AA)

    def _render_net(self, frame: np.ndarray):
        """Render 3D Net with side posts, clamps and top tension cord."""
        nw = self.table_config.half_width + getattr(self.table_config, "NET_OVERHANG", 0.1525)
        nh = getattr(self.table_config, "NET_HEIGHT", 0.1525)

        n_tl = self.project_point(-nw, nh, 0.0)
        n_tr = self.project_point(nw, nh, 0.0)
        n_br = self.project_point(nw, 0.0, 0.0)
        n_bl = self.project_point(-nw, 0.0, 0.0)

        if n_tl and n_tr and n_br and n_bl:
            pts = np.array([[n_tl[0], n_tl[1]], [n_tr[0], n_tr[1]], [n_br[0], n_br[1]], [n_bl[0], n_bl[1]]], np.int32)
            overlay = frame.copy()
            # Semi-transparent dark mesh
            cv2.fillPoly(overlay, [pts], (25, 25, 28))
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Thick white upper tension cord
            cv2.line(frame, (n_tl[0], n_tl[1]), (n_tr[0], n_tr[1]), (250, 250, 255), 3, cv2.LINE_AA)

            # Heavy metal side posts and clamps
            post_color = (65, 68, 75)
            clamp_color = (95, 100, 110)
            cv2.line(frame, (n_bl[0], n_bl[1] + 8), (n_tl[0], n_tl[1] - 4), post_color, 5)
            cv2.line(frame, (n_br[0], n_br[1] + 8), (n_tr[0], n_tr[1] - 4), post_color, 5)
            # Screw knobs
            cv2.circle(frame, (n_bl[0], n_bl[1] + 4), 3, clamp_color, -1)
            cv2.circle(frame, (n_br[0], n_br[1] + 4), 3, clamp_color, -1)

    def _render_landing_target(self, frame: np.ndarray, ball: Ball):
        """Render glowing Wii Sports landing target on table showing where ball will bounce."""
        if not ball.in_play or abs(ball.vz) < 0.2:
            return

        # Simple ballistic approximation for landing spot
        if ball.y > 0.05 and ball.vy < 1.5:
            t_impact = (ball.vy + math.sqrt(max(0.01, ball.vy**2 + 2 * 9.81 * ball.y))) / 9.81
            land_x = ball.x + ball.vx * t_impact
            land_z = ball.z + ball.vz * t_impact

            if abs(land_x) <= self.table_config.half_width and abs(land_z) <= self.table_config.half_length:
                proj = self.project_point(land_x, 0.0, land_z)
                if proj:
                    lx, ly, lz = proj
                    # Countdown affordance (gesture-cv-ux): shrink ring towards bounce
                    countdown = min(2.5, max(0.9, 0.9 + t_impact * 2.5))
                    base_r = int(24.0 / lz)
                    r_outer = max(6, int(base_r * countdown))
                    r_inner = max(2, int(8.0 / lz))

                    # Speed-tiered target colors
                    if ball.speed_kmh > 55.0:
                        color = (40, 40, 255)  # Smash Red BGR
                    elif ball.speed_kmh > 35.0:
                        color = (0, 215, 255)  # Drive Amber BGR
                    else:
                        color = (255, 230, 0) if land_z < 0 else (255, 120, 0)

                    cv2.circle(frame, (lx, ly), r_outer, color, 2)
                    cv2.circle(frame, (lx, ly), r_inner, (255, 255, 255), -1)

    def _render_shadows(self, frame: np.ndarray, ball: Ball):
        """Render soft circular shadow under ball to enhance 3D perception."""
        over_table = (
            abs(ball.x) <= self.table_config.half_width + 0.12
            and abs(ball.z) <= self.table_config.half_length + 0.12
        )
        shadow_y = 0.0 if over_table else -0.76
        shadow_proj = self.project_point(ball.x, shadow_y, ball.z)
        if shadow_proj:
            sx, sy, sz = shadow_proj
            base_r = int(22.0 / sz)
            dist_to_surf = max(0.0, ball.y - shadow_y)
            height_scale = max(0.4, 1.0 - (dist_to_surf * 0.45))
            r = max(3, int(base_r * height_scale))

            overlay = frame.copy()
            cv2.ellipse(overlay, (sx, sy), (r, int(r * 0.45)), 0, 0, 360, (20, 20, 20), -1)
            alpha = max(0.18, min(0.6, 0.7 - dist_to_surf * 0.35))
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _render_ball_trail(self, frame: np.ndarray, ball: Ball):
        """Render speed motion blur trail with velocity and spin heat color grading."""
        spin_mag = math.sqrt(getattr(ball, "spin_x", 0.0) ** 2 + getattr(ball, "spin_y", 0.0) ** 2)
        has_heavy_spin = spin_mag >= 4.0

        for i, pos in enumerate(ball.trail):
            proj = self.project_point(pos[0], pos[1], pos[2])
            if proj:
                alpha = (i + 1) / len(ball.trail)
                r = max(2, int((16.0 / proj[2]) * alpha * 0.75))
                if ball.speed_kmh > 55.0:
                    color = (int(50 * alpha), int(50 * alpha), int(255 * alpha))  # Neon Red BGR
                elif has_heavy_spin and ball.speed_kmh > 20.0:
                    # Dynamic curve/spin trail
                    if getattr(ball, "spin_y", 0.0) > 1.5:
                        color = (int(255 * alpha), int(220 * alpha), 0)  # Cyan curve right
                    elif getattr(ball, "spin_y", 0.0) < -1.5:
                        color = (int(255 * alpha), 0, int(220 * alpha))  # Violet curve left
                    else:
                        color = (0, int(230 * alpha), int(255 * alpha))  # Gold topspin
                elif ball.speed_kmh > 35.0:
                    color = (0, int(200 * alpha), int(255 * alpha))  # Amber BGR
                else:
                    color = (int(255 * alpha), int(230 * alpha), 0)  # Electric Cyan BGR
                cv2.circle(frame, (proj[0], proj[1]), r, color, -1)

    def _render_ball(self, frame: np.ndarray, ball: Ball):
        """Render 3D table tennis ball with spherical shading and squash & stretch."""
        ball_proj = self.project_point(ball.x, ball.y, ball.z)
        if ball_proj:
            bx, by, bz = ball_proj
            base_r = max(4, int(28.0 / bz))

            squash = getattr(ball, "squash", 0.0)
            rx = int(base_r * (1.0 + squash * 0.45))
            ry = max(2, int(base_r * (1.0 - squash * 0.45)))

            # Ball base color (vibrant ITTF orange BGR: 0, 140, 255)
            cv2.ellipse(frame, (bx, by), (rx, ry), 0, 0, 360, (0, 140, 255), -1)
            cv2.ellipse(frame, (bx, by), (rx, ry), 0, 0, 360, (0, 90, 200), 2)

            # Specular highlight
            highlight_x = int(bx - rx * 0.3)
            highlight_y = int(by - ry * 0.3)
            highlight_r = max(2, int(rx * 0.35))
            cv2.circle(frame, (highlight_x, highlight_y), highlight_r, (255, 255, 255), -1)

    def _render_paddle(self, frame: np.ndarray, paddle: Paddle, is_player: bool):
        """Render 3D multilayer paddle with classic Red/Black rubbers, wood core and flared handle."""
        proj = self.project_point(paddle.x, paddle.y, paddle.z)
        if not proj:
            return

        px, py, pz = proj
        vis_r = getattr(paddle.config, "VISUAL_RADIUS", 0.076)
        blade_r = max(10, int((self.focal_length * vis_r) / pz))

        # Dynamic 3D roll and pitch angles
        tilt_angle = math.degrees(getattr(paddle, "roll", getattr(paddle, "tilt_y", 0.0)))
        tilt_rad = math.radians(tilt_angle)
        pitch = getattr(paddle, "pitch", getattr(paddle, "tilt_x", 0.0))

        # Determine Forehand (Red) vs Backhand (Black) face orientation
        p_normal = getattr(paddle, "normal", np.array([0.0, 0.0, 1.0]))
        is_front_face = (p_normal[2] >= 0.0) if is_player else (p_normal[2] <= 0.0)

        # Classic International Combination per user request: Red on front, Black on back
        if is_front_face:
            rubber_color = (25, 28, 215)   # Classic Championship Ruby Red in BGR
            rubber_border = (15, 18, 160)
        else:
            rubber_color = (25, 25, 30)     # Classic Carbon Black in BGR
            rubber_border = (15, 15, 18)

        # 3D Depth Extrusion offset based on blade pitch and roll
        edge_dx = int(math.sin(tilt_rad) * pitch * blade_r * 0.35)
        edge_dy = int(-math.cos(tilt_rad) * pitch * blade_r * 0.35)

        # 1. Flared Ergonomic Handle proportional to ITTF blade
        handle_len = int(blade_r * 0.90)
        w_neck = max(3, int(blade_r * 0.20))
        w_flare = max(4, int(blade_r * 0.30))

        if is_player:
            base_dir_x = math.sin(tilt_rad)
            base_dir_y = math.cos(tilt_rad)
            perp_x = -base_dir_y
            perp_y = base_dir_x

            neck_center_x = px + int(base_dir_x * (blade_r * 0.40))
            neck_center_y = py + int(base_dir_y * (blade_r * 0.40))
            end_center_x = neck_center_x + int(base_dir_x * handle_len)
            end_center_y = neck_center_y + int(base_dir_y * handle_len)

            p_h1 = (neck_center_x - int(perp_x * w_neck), neck_center_y - int(perp_y * w_neck))
            p_h2 = (neck_center_x + int(perp_x * w_neck), neck_center_y + int(perp_y * w_neck))
            p_h3 = (end_center_x + int(perp_x * w_flare), end_center_y + int(perp_y * w_flare))
            p_h4 = (end_center_x - int(perp_x * w_flare), end_center_y - int(perp_y * w_flare))

            h_poly = np.array([p_h1, p_h2, p_h3, p_h4], np.int32)
            cv2.fillPoly(frame, [h_poly], (70, 110, 165))
            cv2.polylines(frame, [h_poly], True, (45, 75, 120), 1, cv2.LINE_AA)

            # Center two-tone grip inlay stripe
            cv2.line(frame, (neck_center_x, neck_center_y), (end_center_x, end_center_y), (50, 75, 110), max(2, w_neck // 2))

            # Metallic emblem jewel badge at base
            emblem_x = end_center_x - int(base_dir_x * (handle_len * 0.18))
            emblem_y = end_center_y - int(base_dir_y * (handle_len * 0.18))
            cv2.circle(frame, (emblem_x, emblem_y), max(2, int(w_flare * 0.35)), (180, 200, 220), -1)
            cv2.circle(frame, (emblem_x, emblem_y), max(2, int(w_flare * 0.35)), (80, 100, 120), 1)
        else:
            handle_end_y = py - handle_len
            cv2.line(frame, (px, py - int(blade_r * 0.4)), (px, handle_end_y), (65, 100, 150), max(4, int(blade_r * 0.28)))

        # 2. Motion Ghosting for Fast Swings
        axes = (blade_r, max(4, int(blade_r * 0.90)))
        if is_player and hasattr(paddle, "trail") and len(paddle.trail) >= 2:
            spd = max(getattr(paddle, "speed", 0.0), getattr(paddle, "hand_speed", 0.0))
            if spd >= 0.8:
                trail_len = len(paddle.trail)
                for idx, (tx, ty, tz) in enumerate(paddle.trail[:-1]):
                    t_proj = self.project_point(tx, ty, tz)
                    if t_proj:
                        t_ratio = (idx + 1) / trail_len
                        ghost_color = (int(255 * t_ratio * 0.35), int(220 * t_ratio * 0.35), int(50 * t_ratio * 0.35))
                        t_axes = (int(axes[0] * (0.85 + 0.15 * t_ratio)), int(axes[1] * (0.85 + 0.15 * t_ratio)))
                        cv2.ellipse(frame, (t_proj[0], t_proj[1]), t_axes, tilt_angle, 0, 360, ghost_color, 1)

        # 3. 3D Blade Edge: Wood Core & Edge Tape
        if abs(edge_dx) + abs(edge_dy) > 1:
            cv2.ellipse(frame, (px + edge_dx, py + edge_dy), axes, tilt_angle, 0, 360, (20, 20, 25), -1)
            cv2.ellipse(frame, (px + edge_dx // 2, py + edge_dy // 2), axes, tilt_angle, 0, 360, (110, 160, 215), -1)

        # Main Blade Surface
        cv2.ellipse(frame, (px, py), axes, tilt_angle, 0, 360, rubber_color, -1)
        cv2.ellipse(frame, (px, py), axes, tilt_angle, 0, 360, rubber_border, max(1, int(blade_r * 0.08)))
        cv2.ellipse(frame, (px, py), (axes[0] + 1, axes[1] + 1), tilt_angle, 0, 360, (50, 180, 225), 1)

        # Specular Sheen on Rubber Surface
        spec_x = int(px - axes[0] * 0.28)
        spec_y = int(py - axes[1] * 0.28)
        spec_axes = (max(2, int(axes[0] * 0.40)), max(2, int(axes[1] * 0.30)))
        spec_overlay = frame.copy()
        cv2.ellipse(spec_overlay, (spec_x, spec_y), spec_axes, tilt_angle, 0, 360, (255, 255, 255), -1)
        cv2.addWeighted(spec_overlay, 0.28, frame, 0.72, 0, frame)

        # Player-specific visual energy & sweet spot cues
        if is_player:
            # Inner sweet-spot target ring
            cv2.ellipse(frame, (px, py), (int(axes[0] * 0.34), int(axes[1] * 0.34)), tilt_angle, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)

            # Swing Power Aura
            spd = max(getattr(paddle, "speed", 0.0), getattr(paddle, "hand_speed", 0.0))
            if spd >= 2.4 or (getattr(paddle, "is_hook_swing", False) and spd >= 1.8):
                cv2.ellipse(frame, (px, py), (int(axes[0] * 1.18), int(axes[1] * 1.18)), tilt_angle, 0, 360, (0, 140, 255), 3)
            elif spd >= 1.0:
                cv2.ellipse(frame, (px, py), (int(axes[0] * 1.12), int(axes[1] * 1.12)), tilt_angle, 0, 360, (255, 230, 0), 2)

        # Label tag above paddle
        tag = "VOCE" if is_player else "ROBERVAL"
        offset_x = 22 if is_player else 38
        cv2.putText(frame, tag, (px - offset_x + 1, py - blade_r - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, tag, (px - offset_x, py - blade_r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    def _render_strike_guide(self, frame: np.ndarray, game_state: GameStateMachine):
        """Render lateral instructions and subtle 3D cues without distracting guideline lines."""
        ball = game_state.ball
        paddle = game_state.player_paddle

        # 1. Glowing target indicator on ball during player serve (pure visual sweet spot, no line)
        if game_state.status == MatchStatus.SERVING and game_state.server == "PLAYER":
            b_proj = self.project_point(ball.x, ball.y, ball.z)
            if b_proj:
                pulse = 0.5 + 0.5 * math.sin(time.time() * 8.0)
                pulse_r = int(22 + 10 * pulse)
                cv2.circle(frame, (b_proj[0], b_proj[1]), pulse_r, (0, 215, 255), 2)

        # 2. Dynamic strike reach and intercept indicator on paddle during rally (no line)
        elif ball.in_play and ball.z < -0.6:
            p_proj = self.project_point(paddle.x, paddle.y, paddle.z)
            if p_proj and ball.vz < 0:
                vis_r = getattr(paddle.config, "VISUAL_RADIUS", 0.076)
                blade_r = max(10, int((self.focal_length * vis_r) / p_proj[2]))
                in_strike = (-1.75 <= ball.z <= -0.75)
                color = (0, 255, 120) if in_strike else (0, 215, 255)
                cv2.circle(frame, (p_proj[0], p_proj[1]), int(blade_r * 1.25), color, 2)

        # 3. Discrete Status Pill strictly on the lateral Top-Left
        cooldown = getattr(game_state, "serve_cooldown", 0.0)
        if cooldown > 0.0 and game_state.status == MatchStatus.SERVING:
            winner = getattr(game_state, "last_point_winner", "PLAYER")
            msg = getattr(game_state, "announcement", "") or f"PONTO {'SEU (+1)' if winner == 'PLAYER' else 'DO ROBERVAL (+1)'}"
            color = (120, 220, 40) if winner == "PLAYER" else (70, 70, 230)
        elif game_state.announcement and game_state.announcement_timer > 0:
            msg = game_state.announcement
            is_smash = "SMASH" in msg or "ACE" in msg
            color = (0, 180, 245) if is_smash else ((120, 220, 40) if ("SEU" in msg or "VOCE" in msg) else (70, 70, 230))
        elif game_state.status == MatchStatus.READY:
            msg = "PREPARAR: FECHE A MAO OU ESPACO"
            color = (0, 180, 245)
        elif game_state.status == MatchStatus.SERVING and game_state.server == "PLAYER":
            msg = "SUA VEZ DE SACAR: GOLPEIE OU ESPACO"
            color = (120, 220, 40)
        elif game_state.status == MatchStatus.SERVING and game_state.server in ("AI", "ROBERVAL"):
            msg = "VEZ DO ROBERVAL: AGUARDANDO SAQUE..."
            color = (230, 140, 60)
        elif game_state.status == MatchStatus.RALLY:
            if ball.in_play and ball.vz < 0 and ball.z < -0.75:
                msg = "ZONA DE GOLPE: ATAQUE COM O BRACO!"
                color = (120, 220, 40)
            elif ball.in_play and ball.vz < 0:
                msg = "BOLA CHEGANDO... PREPARE O GOLPE"
                color = (0, 180, 245)
            else:
                msg = f"RALLY ATIVO ({ball.speed_kmh:.0f} km/h)"
                color = (200, 240, 200)
        elif game_state.status == MatchStatus.GAME_OVER:
            msg = "FIM DE JOGO! FECHE A MAO P/ REINICIAR"
            color = (0, 180, 245)
        else:
            msg = ""
            color = (200, 200, 200)

        if msg:
            font_scale = 0.44
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            max_w = min(440, self.width - 50)
            if tw + 45 > max_w:
                font_scale = 0.38
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            pill_w = min(max_w, max(220, tw + 45))
            pill_x, pill_y = 20, 75
            pill_h = 32

            overlay = frame.copy()
            cv2.rectangle(overlay, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (16, 20, 30), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), color, 1)
            cv2.circle(frame, (pill_x + 13, pill_y + int(pill_h / 2)), 4, color, -1)
            cv2.putText(
                frame,
                msg,
                (pill_x + 26, pill_y + int(pill_h / 2) + int(th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _render_announcement_pop(self, frame: np.ndarray, game_state: GameStateMachine):
        """No-op: Announcements are displayed in the lateral status pill to avoid obstructing the middle."""
        return

    def _render_hud(
        self,
        frame: np.ndarray,
        game_state: GameStateMachine,
        mouse_screen_pos: Optional[Tuple[int, int]] = None,
    ):
        """Broadcast-style TV scorebug with serving pulse, set pips, and hover feedback."""
        w = self.width
        cx = int(w / 2)
        sb_w, sb_h = 430, 44
        sb_x = cx - int(sb_w / 2)
        sb_y = 14

        # Drop shadow
        shadow = frame.copy()
        cv2.rectangle(shadow, (sb_x + 2, sb_y + 2), (sb_x + sb_w + 2, sb_y + sb_h + 2), (10, 10, 15), -1)
        cv2.addWeighted(shadow, 0.40, frame, 0.60, 0, frame)

        # Glass container
        overlay = frame.copy()
        cv2.rectangle(overlay, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h), (16, 20, 28), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h), (60, 75, 95), 1)

        # Left Pod: Player
        p_pod_w = 165
        cv2.rectangle(frame, (sb_x, sb_y), (sb_x + 4, sb_y + sb_h), (255, 215, 0), -1)
        cv2.putText(frame, "VOCE", (sb_x + 14, sb_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.54, (255, 255, 255), 1)

        if game_state.server == "PLAYER":
            pulse = 0.5 + 0.5 * math.sin(time.time() * 8.0)
            glow_r = int(6 + 2 * pulse)
            cv2.circle(frame, (sb_x + 68, sb_y + 23), glow_r, (0, 215, 255), -1)
            cv2.circle(frame, (sb_x + 68, sb_y + 23), 7, (255, 255, 255), 1)

        # Sets won pips for Player (best of 3: 2 dots)
        p_sets = getattr(game_state, "player_sets", 0)
        for i in range(2):
            pip_x = sb_x + 14 + i * 11
            pip_y = sb_y + 36
            if i < p_sets:
                cv2.circle(frame, (pip_x, pip_y), 3, (0, 230, 255), -1)
            else:
                cv2.circle(frame, (pip_x, pip_y), 3, (70, 85, 110), 1)

        cv2.rectangle(frame, (sb_x + p_pod_w - 42, sb_y + 4), (sb_x + p_pod_w - 4, sb_y + sb_h - 4), (30, 42, 58), -1)
        cv2.rectangle(frame, (sb_x + p_pod_w - 42, sb_y + 4), (sb_x + p_pod_w - 4, sb_y + sb_h - 4), (0, 200, 255), 1)
        p_score_str = str(game_state.player_score)
        p_score_x = sb_x + p_pod_w - 30 if len(p_score_str) == 1 else sb_x + p_pod_w - 36
        cv2.putText(frame, p_score_str, (p_score_x, sb_y + 31), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 235, 255), 2)

        # Center Pod: Rally Counter with Flame Glow on Streaks
        c_pod_x = sb_x + p_pod_w
        cv2.line(frame, (c_pod_x, sb_y + 6), (c_pod_x, sb_y + sb_h - 6), (45, 55, 70), 1)
        is_fire = game_state.current_rally_count >= 5
        rally_color = (0, 140, 255) if is_fire else (0, 215, 255)
        cv2.putText(frame, "RALLY" if not is_fire else "FIRE!", (c_pod_x + 28, sb_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, rally_color, 1)

        rally_str = str(game_state.current_rally_count)
        r_text_x = c_pod_x + 44 if len(rally_str) == 1 else c_pod_x + 38
        val_color = (0, 180, 255) if is_fire else (255, 255, 255)
        cv2.putText(frame, rally_str, (r_text_x, sb_y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.58, val_color, 1)

        c_pod_end = c_pod_x + 100
        cv2.line(frame, (c_pod_end, sb_y + 6), (c_pod_end, sb_y + sb_h - 6), (45, 55, 70), 1)

        # Right Pod: Roberval
        r_pod_x = c_pod_end
        cv2.rectangle(frame, (r_pod_x + 4, sb_y + 4), (r_pod_x + 42, sb_y + sb_h - 4), (50, 28, 34), -1)
        cv2.rectangle(frame, (r_pod_x + 4, sb_y + 4), (r_pod_x + 42, sb_y + sb_h - 4), (255, 90, 100), 1)
        r_score_str = str(game_state.ai_score)
        r_score_x = r_pod_x + 16 if len(r_score_str) == 1 else r_pod_x + 10
        cv2.putText(frame, r_score_str, (r_score_x, sb_y + 31), cv2.FONT_HERSHEY_DUPLEX, 0.75, (120, 140, 255), 2)

        if game_state.server in ("AI", "ROBERVAL"):
            pulse = 0.5 + 0.5 * math.sin(time.time() * 8.0)
            glow_r = int(6 + 2 * pulse)
            cv2.circle(frame, (r_pod_x + 54, sb_y + 23), glow_r, (0, 215, 255), -1)
            cv2.circle(frame, (r_pod_x + 54, sb_y + 23), 7, (255, 255, 255), 1)

        # Sets won pips for Roberval
        a_sets = getattr(game_state, "ai_sets", 0)
        for i in range(2):
            pip_x = r_pod_x + 68 + i * 11
            pip_y = sb_y + 36
            if i < a_sets:
                cv2.circle(frame, (pip_x, pip_y), 3, (60, 60, 240), -1)
            else:
                cv2.circle(frame, (pip_x, pip_y), 3, (70, 85, 110), 1)

        r_name_x = r_pod_x + 68 if game_state.server in ("AI", "ROBERVAL") else r_pod_x + 52
        cv2.putText(frame, "ROBERVAL", (r_name_x, sb_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1)
        cv2.rectangle(frame, (sb_x + sb_w - 4, sb_y), (sb_x + sb_w, sb_y + sb_h), (60, 60, 235), -1)

        # Discrete Match Difficulty Tag below scorebug
        diff = getattr(game_state, "difficulty", "MEDIUM")
        cv2.putText(frame, f"DIFICULDADE: {diff}", (sb_x + 4, sb_y + sb_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 160, 180), 1)

        # Center below scorebug: Current Set & Score
        p_sets = getattr(game_state, "player_sets", 0)
        a_sets = getattr(game_state, "ai_sets", 0)
        s_num = getattr(game_state, "set_number", 1)
        set_str = f"SET {s_num}  |  SETS: {p_sets} x {a_sets}"
        cv2.putText(frame, set_str, (cx - 72, sb_y + sb_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 220, 240), 1)

        # Real-time Ball Speedometer during rally
        if game_state.ball.in_play and game_state.status == MatchStatus.RALLY and game_state.ball.speed_kmh > 4:
            b_spd = game_state.ball.speed_kmh
            spd_lbl = f"BOLA: {b_spd:.0f} KM/H"
            spd_color = (0, 140, 255) if b_spd > 50 else ((0, 220, 255) if b_spd > 35 else (180, 240, 180))
            cv2.putText(frame, spd_lbl, (sb_x + sb_w - 105, sb_y + sb_h + 16), cv2.FONT_HERSHEY_DUPLEX, 0.40, spd_color, 1)

        # In-game HUD Menu Button (top-left) with hover
        hx1, hy1, hx2, hy2 = self.hud_menu_button
        is_hover_hud = False
        if mouse_screen_pos is not None:
            mx, my = mouse_screen_pos
            is_hover_hud = (hx1 <= mx <= hx2 and hy1 <= my <= hy2)

        overlay = frame.copy()
        bg_col = (55, 80, 115) if is_hover_hud else (18, 22, 32)
        cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), bg_col, -1)
        cv2.addWeighted(overlay, 0.90 if is_hover_hud else 0.85, frame, 0.10 if is_hover_hud else 0.15, 0, frame)
        border_col = (0, 230, 255) if is_hover_hud else (55, 70, 95)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), border_col, 2 if is_hover_hud else 1)
        txt = "▸ [M] MENU" if is_hover_hud else "[M] MENU"
        cv2.putText(frame, txt, (hx1 + (6 if is_hover_hud else 10), hy1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255) if is_hover_hud else (210, 225, 240), 1, cv2.LINE_AA)

    def get_menu_action_at(self, x: int, y: int) -> Optional[str]:
        """Return the menu action identifier for a screen coordinate (x, y), or None."""
        for name, (bx1, by1, bx2, by2) in self.menu_buttons.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                return name
        return None

    def is_hud_menu_clicked(self, x: int, y: int) -> bool:
        """Check if in-game HUD Menu button was clicked."""
        hx1, hy1, hx2, hy2 = self.hud_menu_button
        return hx1 <= x <= hx2 and hy1 <= y <= hy2

    def _render_menu_modal(
        self,
        frame: np.ndarray,
        game_state: GameStateMachine,
        track_res: Optional[object] = None,
        in_menu: bool = False,
        mouse_screen_pos: Optional[Tuple[int, int]] = None,
    ):
        """Render modern glassmorphic sports modal for Start Menu / Lobby, Pause Menu, and Set End."""
        w, h = self.width, self.height

        # Reset clickable hitboxes for this frame
        self.menu_buttons.clear()

        # 1. Dark semi-transparent backdrop to focus attention
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (8, 10, 16), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # 2. Main Modal Card (placed on left-to-center to harmonize with right-side live camera)
        mw, mh = 680, 484
        x1 = 30
        y1 = 28
        x2 = x1 + mw
        y2 = y1 + mh
        cx = x1 + int(mw / 2)

        card = frame.copy()
        cv2.rectangle(card, (x1, y1), (x2, y2), (14, 18, 28), -1)
        cv2.addWeighted(card, 0.95, frame, 0.05, 0, frame)

        is_game_over = (game_state.status == MatchStatus.GAME_OVER)
        player_won = is_game_over and (game_state.player_score > game_state.ai_score)
        is_paused = in_menu and (game_state.player_score > 0 or game_state.ai_score > 0 or game_state.status == MatchStatus.RALLY)

        if is_game_over:
            if player_won:
                theme_color = (0, 215, 255)  # Gold
                header_title = "VOCE VENCEU O SET!"
            else:
                theme_color = (80, 80, 240)  # Coral Red
                header_title = "ROBERVAL VENCEU O SET"
            subtitle = f"PLACAR FINAL: {game_state.player_score} x {game_state.ai_score}   |   SETS: {getattr(game_state, 'player_sets', 0)} x {getattr(game_state, 'ai_sets', 0)}"
            action_label = "[ ▶ ]  INICIAR PROXIMO SET"
        elif is_paused:
            theme_color = (0, 230, 255)
            header_title = "JOGO PAUSADO - MENU"
            subtitle = f"PLACAR ATUAL: VOCE {game_state.player_score} x {game_state.ai_score} ROBERVAL   |   SET {getattr(game_state, 'set_number', 1)}"
            action_label = "[ ▶ ]  CONTINUAR PARTIDA"
        else:
            theme_color = (0, 230, 255)  # Neon Cyan
            header_title = "AIRPONG - COPA ROBERVAL 2026"
            subtitle = "CAMPEONATO DE TENIS DE MESA | BIOMECANICA 60 FPS"
            action_label = "[ ▶ ]  INICIAR PARTIDA"

        # Outer card borders with accent glow
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 68, 92), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 4), theme_color, -1)

        # 3. Header Texts
        (tw, th), _ = cv2.getTextSize(header_title, cv2.FONT_HERSHEY_DUPLEX, 0.88, 2)
        cv2.putText(frame, header_title, (cx - int(tw / 2), y1 + 36), cv2.FONT_HERSHEY_DUPLEX, 0.88, (255, 255, 255), 2, cv2.LINE_AA)

        (sw, sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, subtitle, (cx - int(sw / 2), y1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, theme_color, 1, cv2.LINE_AA)

        # Divider line
        cv2.line(frame, (x1 + 25, y1 + 70), (x2 - 25, y1 + 70), (40, 52, 72), 1)

        # 4. Live Biomechanics & Grip Status Pod
        pod_x = x1 + 25
        pod_y = y1 + 78
        pod_w = mw - 50
        pod_h = 76

        cv2.rectangle(frame, (pod_x, pod_y), (pod_x + pod_w, pod_y + pod_h), (20, 26, 40), -1)
        cv2.rectangle(frame, (pod_x, pod_y), (pod_x + pod_w, pod_y + pod_h), (48, 62, 85), 1)

        has_hand = track_res is not None and getattr(track_res, "detected", False)
        if has_hand:
            grip_code = getattr(track_res, "grip_type", "CLASSICO")
            is_grip = getattr(track_res, "is_fist", False) or getattr(track_res, "is_grip", False)
            elbow_deg = getattr(track_res, "elbow_angle", 0.0)
            arm_ok = getattr(track_res, "arm_detected", False)
            hand_side = getattr(track_res, "playing_hand", "direita").upper()

            grip_names = {
                "CLASSICO": "CLASSICO (SHAKEHAND)",
                "CANETA": "CANETA JAPONESA",
                "CLASSINETA": "CLASSINETA",
                "FIST": "PUNHO FECHADO",
                "OPEN": "MAO ABERTA",
            }
            display_grip = grip_names.get(grip_code, grip_code)

            if is_grip:
                status_color = (60, 240, 120)  # Neon Green
                status_dot_color = (60, 240, 120)
                status_title = f"RAQUETE EMPUNHADA: {display_grip}  [PRONTO]"
                status_desc = "Mao fechada detectada! Raquete virtual anexada a sua empunhadura."
            else:
                status_color = (0, 215, 255)  # Gold/Yellow
                status_dot_color = (0, 215, 255)
                status_title = f"MAO DETECTADA: {display_grip}  [AGUARDANDO EMPUNHADURA]"
                status_desc = "Feche a mao como se segurasse a raquete (Classico, Caneta ou Classineta)."

            arm_name = getattr(track_res, "playing_arm", hand_side)
            both_sh = getattr(track_res, "both_shoulders_detected", False)
            sh_status = "Ombros: Ambos OK [x2]" if both_sh else "Ombro: OK"
            arm_str = f"Cotovelo: {int(elbow_deg)}deg" if arm_ok else "Braco: Rastreando"
            meta_str = f"Braco Ativo: {arm_name}  |  {sh_status}  |  {arm_str}  |  Vel: {getattr(track_res, 'speed_kmh', 0.0):.0f} km/h"
        elif track_res is not None and (getattr(track_res, "both_shoulders_detected", False) or getattr(track_res, "arm_detected", False)):
            status_color = (0, 215, 255)  # Gold
            status_dot_color = (0, 215, 255)
            arm_name = getattr(track_res, "playing_arm", "DIREITO")
            status_title = f"CORPO & OMBROS RECONHECIDOS [BRACO {arm_name}]"
            status_desc = "Ombros detectados! Agora levante e feche a mao para empunhar a raquete."
            meta_str = f"Braco: {arm_name}  |  Ombros: Ambos OK  |  Cotovelo: {int(getattr(track_res, 'elbow_angle', 120))}deg"
        else:
            status_color = (80, 100, 255)  # Light Red
            status_dot_color = (70, 70, 240)
            status_title = "AGUARDANDO JOGADOR EM FRENTE A CAMERA"
            status_desc = "Posicione sua mao em frente a webcam para empunhar a raquete virtual."
            meta_str = "Controle alternativo por MOUSE tambem ativo (mova o cursor para rebater)"

        # Status Dot
        cv2.circle(frame, (pod_x + 18, pod_y + 20), 6, status_dot_color, -1)
        cv2.circle(frame, (pod_x + 18, pod_y + 20), 8, (255, 255, 255), 1)

        cv2.putText(frame, status_title, (pod_x + 34, pod_y + 24), cv2.FONT_HERSHEY_DUPLEX, 0.44, status_color, 1, cv2.LINE_AA)
        cv2.putText(frame, status_desc, (pod_x + 34, pod_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (215, 225, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, meta_str, (pod_x + 34, pod_y + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 160, 180), 1, cv2.LINE_AA)

        # 5. Primary Action Button (Pulsing Glow & Hover)
        btn1_w, btn1_h = pod_w, 48
        btn1_x = pod_x
        btn1_y = pod_y + pod_h + 12

        is_hover_start = False
        if mouse_screen_pos is not None:
            mx, my = mouse_screen_pos
            is_hover_start = (btn1_x <= mx <= btn1_x + btn1_w and btn1_y <= my <= btn1_y + btn1_h)

        pulse = 0.5 + 0.5 * math.sin(time.time() * 5.0)
        if is_hover_start:
            btn1_bg = (180, 110, 0)  # Neon Cyan Hover Fill (BGR)
            btn1_border = (0, 255, 255)
        else:
            btn1_bg = (int(26 + 18 * pulse), int(42 + 24 * pulse), int(64 + 30 * pulse))
            btn1_border = theme_color

        cv2.rectangle(frame, (btn1_x, btn1_y), (btn1_x + btn1_w, btn1_y + btn1_h), btn1_bg, -1)
        cv2.rectangle(frame, (btn1_x, btn1_y), (btn1_x + btn1_w, btn1_y + btn1_h), btn1_border, 2)

        lbl = f"▸ {action_label}" if is_hover_start else action_label
        (aw, ah), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        cv2.putText(frame, lbl, (cx - int(aw / 2), btn1_y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        sub_action = "Pressione [ESPACO] ou [ENTER] ou clique aqui com o mouse para iniciar"
        (saw, sah), _ = cv2.getTextSize(sub_action, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
        cv2.putText(frame, sub_action, (cx - int(saw / 2), btn1_y + 41), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255) if is_hover_start else (180, 220, 255), 1, cv2.LINE_AA)

        self.menu_buttons["start"] = (btn1_x, btn1_y, btn1_x + btn1_w, btn1_y + btn1_h)

        # 6. Difficulty Selector Row with Hover
        diff_y = btn1_y + btn1_h + 12
        diff_str = getattr(game_state, "difficulty", "MEDIUM")
        levels = [
            ("1", "FACIL", "EASY", "Iniciante"),
            ("2", "MEDIO", "MEDIUM", "Equilibrado"),
            ("3", "DIFICIL", "HARD", "Desafiador"),
        ]

        lbl_diff = "DIFICULDADE DA IA:"
        cv2.putText(frame, lbl_diff, (pod_x + 4, diff_y + 12), cv2.FONT_HERSHEY_DUPLEX, 0.38, (160, 180, 205), 1, cv2.LINE_AA)

        bw = int((pod_w - 24) / 3)
        bh = 38
        box_start_x = pod_x
        box_y = diff_y + 18

        for i, (k, name, d_val, sub) in enumerate(levels):
            bx = box_start_x + i * (bw + 12)
            is_active = (diff_str == d_val)
            is_hover_d = False
            if mouse_screen_pos is not None:
                mx, my = mouse_screen_pos
                is_hover_d = (bx <= mx <= bx + bw and box_y <= my <= box_y + bh)

            if is_hover_d:
                bg_c = (70, 95, 135)
                border_c = (0, 255, 255)
                text_c = (255, 255, 255)
            elif is_active:
                bg_c = (55, 80, 115)
                border_c = (0, 230, 255)
                text_c = (255, 255, 255)
            else:
                bg_c = (22, 28, 40)
                border_c = (45, 58, 76)
                text_c = (150, 168, 190)

            cv2.rectangle(frame, (bx, box_y), (bx + bw, box_y + bh), bg_c, -1)
            cv2.rectangle(frame, (bx, box_y), (bx + bw, box_y + bh), border_c, 2 if (is_active or is_hover_d) else 1)
            item_text = f"▸ [{k}] {name}" if is_hover_d else f"[{k}] {name}"
            cv2.putText(frame, item_text, (bx + 8, box_y + 17), cv2.FONT_HERSHEY_DUPLEX, 0.36, text_c, 1, cv2.LINE_AA)
            cv2.putText(frame, sub, (bx + 8, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 210, 235) if (is_active or is_hover_d) else (100, 115, 135), 1, cv2.LINE_AA)

            self.menu_buttons[f"diff_{k}"] = (bx, box_y, bx + bw, box_y + bh)

        # 7. Secondary Action Buttons Row with Hover
        sec_y = box_y + bh + 12
        sec_w = int((pod_w - 14) / 2)
        sec_h = 32

        # Left: Restart Match
        r_x = pod_x
        is_hover_r = (mouse_screen_pos is not None and r_x <= mouse_screen_pos[0] <= r_x + sec_w and sec_y <= mouse_screen_pos[1] <= sec_y + sec_h)
        r_bg = (35, 48, 68) if is_hover_r else (24, 30, 44)
        r_border = (0, 230, 255) if is_hover_r else (55, 70, 95)
        cv2.rectangle(frame, (r_x, sec_y), (r_x + sec_w, sec_y + sec_h), r_bg, -1)
        cv2.rectangle(frame, (r_x, sec_y), (r_x + sec_w, sec_y + sec_h), r_border, 2 if is_hover_r else 1)
        r_label = "▸ [ R ] REINICIAR (0 x 0)" if is_hover_r else "[ R ]  REINICIAR PLACAR (0 x 0)"
        (rw, rh), _ = cv2.getTextSize(r_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(frame, r_label, (r_x + int((sec_w - rw) / 2), sec_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255) if is_hover_r else (210, 225, 240), 1, cv2.LINE_AA)
        self.menu_buttons["restart"] = (r_x, sec_y, r_x + sec_w, sec_y + sec_h)

        # Right: Calibration / Training
        t_x = r_x + sec_w + 14
        is_hover_t = (mouse_screen_pos is not None and t_x <= mouse_screen_pos[0] <= t_x + sec_w and sec_y <= mouse_screen_pos[1] <= sec_y + sec_h)
        t_bg = (35, 48, 68) if is_hover_t else (24, 30, 44)
        t_border = (0, 230, 255) if is_hover_t else (55, 70, 95)
        cv2.rectangle(frame, (t_x, sec_y), (t_x + sec_w, sec_y + sec_h), t_bg, -1)
        cv2.rectangle(frame, (t_x, sec_y), (t_x + sec_w, sec_y + sec_h), t_border, 2 if is_hover_t else 1)
        t_label = "▸ [ T ] CALIBRAR BRACO" if is_hover_t else "[ T ]  CALIBRAR MOVIMENTOS DO BRACO"
        (tw2, th2), _ = cv2.getTextSize(t_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(frame, t_label, (t_x + int((sec_w - tw2) / 2), sec_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255) if is_hover_t else (210, 225, 240), 1, cv2.LINE_AA)
        self.menu_buttons["calibrate"] = (t_x, sec_y, t_x + sec_w, sec_y + sec_h)

        # 8. Instructions & Pro Tips Box
        tip_y = sec_y + sec_h + 10
        tip_w = pod_w
        tip_h = 44
        cv2.rectangle(frame, (pod_x, tip_y), (pod_x + tip_w, tip_y + tip_h), (16, 20, 30), -1)
        cv2.rectangle(frame, (pod_x, tip_y), (pod_x + tip_w, tip_y + tip_h), (40, 50, 68), 1)

        tip_title = "COMO JOGAR: 1. Segure a mao fechada como se estivesse com a raquete fisica na mao."
        tip_sub = "2. Mova o braco para frente para rebater. Acelere o antebraco para aplicar Topspin e Smash!"
        cv2.putText(frame, tip_title, (pod_x + 12, tip_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 215, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, tip_sub, (pod_x + 12, tip_y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 175, 195), 1, cv2.LINE_AA)

        # 9. Modal Footer
        footer_text = "[M] MENU  |  [F] TELA CHEIA  |  [Z] PROFUNDIDADE 3D  |  [Q / ESC] SAIR"
        (fw, fh), _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        cv2.putText(frame, footer_text, (cx - int(fw / 2), y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 138, 160), 1, cv2.LINE_AA)
