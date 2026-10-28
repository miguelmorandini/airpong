"""3D Perspective Projection Renderer with Wii Sports table tennis aesthetics."""

import math
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

        # Camera placement: Wii Sports 3D perspective looking down at table
        self.cam_x = 0.0
        self.cam_y = 1.25
        self.cam_z = -3.50
        self.cam_pitch = math.radians(-18.0)  # Looking down 18 degrees
        self.fov = math.radians(48.0)
        self.focal_length = (self.width / 2.0) / math.tan(self.fov / 2.0)

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

        return screen_x, screen_y, cam_z

    def render_frame(self, game_state: GameStateMachine) -> np.ndarray:
        """Render complete game scene to BGR image buffer."""
        # 1. Background (Gradient gymnasium/stadium sky)
        frame = self._render_background()

        # 2. Gymnasium Floor
        self._render_floor(frame)

        # 3. 3D Table Surface & Markings
        self._render_table(frame)

        # 4. 3D Net
        self._render_net(frame)

        # 4.1. Landing Target & Anticipation Reticle
        self._render_landing_target(frame, game_state.ball)

        # 5. Shadows
        self._render_shadows(frame, game_state.ball)

        # 6. Motion Trails
        self._render_ball_trail(frame, game_state.ball)

        # 7. AI Paddle
        self._render_paddle(frame, game_state.ai_paddle, is_player=False)

        # 8. 3D Ball
        self._render_ball(frame, game_state.ball)

        # 9. Player Paddle (Rendered in front)
        self._render_paddle(frame, game_state.player_paddle, is_player=True)

        # 9.1. Strike Guide & Hitting Assist Prompt
        self._render_strike_guide(frame, game_state)

        # 10. Wii Sports HUD & Announcements
        self._render_hud(frame, game_state)

        return frame

    def _render_background(self) -> np.ndarray:
        """Generate smooth vertical gradient background."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Wii Sports Cyan/Sky gradient
        top_color = np.array([245, 215, 175], dtype=np.float32)     # BGR
        bot_color = np.array([215, 175, 120], dtype=np.float32)

        for y in range(self.height):
            ratio = y / self.height
            color = (1.0 - ratio) * top_color + ratio * bot_color
            frame[y, :] = color.astype(np.uint8)
        return frame

    def _render_floor(self, frame: np.ndarray):
        """Render wooden stadium floor with court lines."""
        # Project floor quad
        p1 = self.project_point(-5.0, -0.76, -3.2)
        p2 = self.project_point(5.0, -0.76, -3.2)
        p3 = self.project_point(5.0, -0.76, 4.5)
        p4 = self.project_point(-5.0, -0.76, 4.5)

        if p1 and p2 and p3 and p4:
            pts = np.array([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]], np.int32)
            cv2.fillPoly(frame, [pts], (140, 165, 195))  # Warm wooden tone in BGR
            # Border court line
            cv2.polylines(frame, [pts], True, (255, 255, 255), 2)

    def _render_table(self, frame: np.ndarray):
        """Render 3D ITTF table top, white edges, and center line."""
        w2 = self.table_config.half_width
        l2 = self.table_config.half_length
        y = self.table_config.SURFACE_Y

        # Table surface corners
        c_nw = self.project_point(-w2, y, l2)   # AI left
        c_ne = self.project_point(w2, y, l2)    # AI right
        c_se = self.project_point(w2, y, -l2)   # Player right
        c_sw = self.project_point(-w2, y, -l2)  # Player left

        if c_nw and c_ne and c_se and c_sw:
            # Table top quad
            table_pts = np.array([
                [c_sw[0], c_sw[1]],
                [c_se[0], c_se[1]],
                [c_ne[0], c_ne[1]],
                [c_nw[0], c_nw[1]],
            ], np.int32)

            # Classic ITTF Deep Blue table surface
            cv2.fillPoly(frame, [table_pts], (165, 90, 15))

            # Table side apron thickness
            apron_y = y - 0.04
            c_se_b = self.project_point(w2, apron_y, -l2)
            c_sw_b = self.project_point(-w2, apron_y, -l2)
            if c_se_b and c_sw_b:
                front_apron = np.array([
                    [c_sw[0], c_sw[1]],
                    [c_se[0], c_se[1]],
                    [c_se_b[0], c_se_b[1]],
                    [c_sw_b[0], c_sw_b[1]],
                ], np.int32)
                cv2.fillPoly(frame, [front_apron], (115, 60, 10))

            # Crisp white border lines
            cv2.polylines(frame, [table_pts], True, (250, 250, 250), 3)

            # White center dividing line along Z
            m_near = self.project_point(0.0, y, -l2)
            m_far = self.project_point(0.0, y, l2)
            if m_near and m_far:
                cv2.line(frame, (m_near[0], m_near[1]), (m_far[0], m_far[1]), (250, 250, 250), 2)

    def _render_net(self, frame: np.ndarray):
        """Render 3D table tennis net with mesh and top cord."""
        overhang = self.table_config.half_width + self.table_config.NET_OVERHANG
        ny = self.table_config.SURFACE_Y + self.table_config.NET_HEIGHT

        # Net 4 corners at Z = 0
        n_bl = self.project_point(-overhang, 0.0, 0.0)
        n_br = self.project_point(overhang, 0.0, 0.0)
        n_tr = self.project_point(overhang, ny, 0.0)
        n_tl = self.project_point(-overhang, ny, 0.0)

        if n_bl and n_br and n_tr and n_tl:
            net_pts = np.array([
                [n_bl[0], n_bl[1]],
                [n_br[0], n_br[1]],
                [n_tr[0], n_tr[1]],
                [n_tl[0], n_tl[1]],
            ], np.int32)

            # Semi-transparent dark mesh
            overlay = frame.copy()
            cv2.fillPoly(overlay, [net_pts], (60, 60, 60))
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # White top tape
            cv2.line(frame, (n_tl[0], n_tl[1]), (n_tr[0], n_tr[1]), (255, 255, 255), 3)
            # Net posts
            cv2.line(frame, (n_bl[0], n_bl[1]), (n_tl[0], n_tl[1]), (50, 50, 50), 4)
            cv2.line(frame, (n_br[0], n_br[1]), (n_tr[0], n_tr[1]), (50, 50, 50), 4)

    def _render_landing_target(self, frame: np.ndarray, ball: Ball):
        """Render glowing Wii Sports landing target on table showing where ball will bounce."""
        if not ball.in_play or abs(ball.vz) < 0.2:
            return

        # Estimate where ball will cross Y=0
        # Simple ballistic approximation for landing spot
        if ball.y > 0.05 and ball.vy < 1.5:
            t_impact = (ball.vy + math.sqrt(max(0.01, ball.vy**2 + 2 * 9.81 * ball.y))) / 9.81
            land_x = ball.x + ball.vx * t_impact
            land_z = ball.z + ball.vz * t_impact

            # Check if landing is on table
            if abs(land_x) <= self.table_config.half_width and abs(land_z) <= self.table_config.half_length:
                proj = self.project_point(land_x, 0.0, land_z)
                if proj:
                    lx, ly, lz = proj
                    r_outer = max(6, int(35.0 / lz))
                    r_inner = max(2, int(10.0 / lz))

                    # Pulsing target rings
                    color = (0, 240, 255) if land_z < 0 else (255, 120, 0)
                    cv2.circle(frame, (lx, ly), r_outer, color, 2)
                    cv2.circle(frame, (lx, ly), r_inner, (255, 255, 255), -1)

    def _render_shadows(self, frame: np.ndarray, ball: Ball):
        """Render soft circular shadow under ball to enhance 3D perception."""
        # Shadow on table surface (Y = 0)
        shadow_proj = self.project_point(ball.x, 0.0, ball.z)
        if shadow_proj:
            sx, sy, sz = shadow_proj
            # Scale shadow by ball height (higher ball = larger, dimmer shadow)
            base_r = int(22.0 / sz)
            height_scale = max(0.5, 1.0 - (ball.y * 0.4))
            r = max(3, int(base_r * height_scale))

            # Draw translucent shadow ellipse
            overlay = frame.copy()
            cv2.ellipse(overlay, (sx, sy), (r, int(r * 0.45)), 0, 0, 360, (20, 20, 20), -1)
            alpha = max(0.2, min(0.6, 0.7 - ball.y * 0.3))
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _render_ball_trail(self, frame: np.ndarray, ball: Ball):
        """Render speed motion blur trail."""
        for i, pos in enumerate(ball.trail):
            proj = self.project_point(pos[0], pos[1], pos[2])
            if proj:
                alpha = (i + 1) / len(ball.trail)
                r = max(2, int((16.0 / proj[2]) * alpha * 0.7))
                color = (0, int(160 * alpha), int(255 * alpha))  # Vibrant yellow-orange
                cv2.circle(frame, (proj[0], proj[1]), r, color, -1)

    def _render_ball(self, frame: np.ndarray, ball: Ball):
        """Render 3D table tennis ball with 3D spherical shading."""
        ball_proj = self.project_point(ball.x, ball.y, ball.z)
        if ball_proj:
            bx, by, bz = ball_proj
            # Apparent radius scales with distance
            radius = max(4, int(28.0 / bz))

            # Ball base color (vibrant ITTF orange BGR: 0, 140, 255)
            cv2.circle(frame, (bx, by), radius, (0, 140, 255), -1)
            # Edge shading
            cv2.circle(frame, (bx, by), radius, (0, 90, 200), 2)
            # Specular light highlight
            highlight_x = int(bx - radius * 0.3)
            highlight_y = int(by - radius * 0.3)
            highlight_r = max(2, int(radius * 0.35))
            cv2.circle(frame, (highlight_x, highlight_y), highlight_r, (255, 255, 255), -1)

    def _render_paddle(self, frame: np.ndarray, paddle: Paddle, is_player: bool):
        """Render 3D paddle with rubber face, border rim and wooden handle."""
        proj = self.project_point(paddle.x, paddle.y, paddle.z)
        if not proj:
            return

        px, py, pz = proj
        blade_r = max(8, int(60.0 / pz))

        # Color: Player has Red rubber, AI has Black rubber
        rubber_color = (35, 35, 225) if is_player else (40, 40, 40)
        rim_color = (240, 240, 240) if is_player else (180, 180, 180)
        handle_color = (90, 120, 160)  # Wood

        # Handle
        handle_len = int(blade_r * 1.0)
        handle_w = max(4, int(blade_r * 0.28))
        handle_end_y = py + handle_len if is_player else py - handle_len
        cv2.line(frame, (px, py + int(blade_r * 0.4)), (px, handle_end_y), handle_color, handle_w)

        # Paddle blade circular disc
        cv2.circle(frame, (px, py), blade_r, rubber_color, -1)
        cv2.circle(frame, (px, py), blade_r, rim_color, max(1, int(blade_r * 0.08)))

        # Label tag above paddle
        tag = "VOCE" if is_player else "ROBERVAL"
        offset_x = 20 if is_player else 38
        cv2.putText(
            frame,
            tag,
            (px - offset_x, py - blade_r - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    def _render_strike_guide(self, frame: np.ndarray, game_state: GameStateMachine):
        """Render discrete corner status pill and visual guideline without obscuring paddle."""
        ball = game_state.ball
        paddle = game_state.player_paddle

        # 1. Subtle 3D guideline between paddle and ball (NO text blocking paddle)
        if game_state.status == MatchStatus.SERVING and game_state.server == "PLAYER":
            p_proj = self.project_point(paddle.x, paddle.y, paddle.z)
            b_proj = self.project_point(ball.x, ball.y, ball.z)
            if p_proj and b_proj:
                cv2.line(frame, (p_proj[0], p_proj[1]), (b_proj[0], b_proj[1]), (0, 255, 255), 2)
        elif ball.in_play and ball.z < -0.6:
            p_proj = self.project_point(paddle.x, paddle.y, paddle.z)
            b_proj = self.project_point(ball.x, ball.y, ball.z)
            if p_proj and b_proj:
                dx = ball.x - paddle.x
                dy = ball.y - paddle.y
                dist_xy = math.sqrt(dx * dx + dy * dy)
                is_in_reach = dist_xy <= paddle.config.ASSISTED_RADIUS
                line_color = (0, 255, 120) if is_in_reach else (60, 60, 255)
                cv2.line(frame, (p_proj[0], p_proj[1]), (b_proj[0], b_proj[1]), line_color, 2)

        # 2. Discrete Status & Instruction Pill in Top-Left Corner (clean, fixed, never blocks play)
        if game_state.announcement and game_state.announcement_timer > 0:
            msg = game_state.announcement
            color = (0, 230, 255)
        elif game_state.status == MatchStatus.READY:
            msg = "FECHE A MAO PARA INICIAR"
            color = (0, 230, 255)
        elif game_state.status == MatchStatus.SERVING and game_state.server == "PLAYER":
            msg = "SAQUE: GOLPEIE COM FORCA DO BRACO!"
            color = (0, 255, 120)
        elif game_state.status == MatchStatus.SERVING and game_state.server in ("AI", "ROBERVAL"):
            msg = "SAQUE DO ROBERVAL..."
            color = (180, 180, 255)
        elif game_state.status == MatchStatus.RALLY:
            if ball.in_play and ball.z < -0.8:
                dx = ball.x - paddle.x
                dy = ball.y - paddle.y
                dist_xy = math.sqrt(dx * dx + dy * dy)
                is_in_reach = dist_xy <= paddle.config.ASSISTED_RADIUS
                msg = "REBATA AGORA!" if is_in_reach else "ALINHE A RAQUETE COM A BOLA"
                color = (0, 255, 255) if is_in_reach else (100, 200, 255)
            else:
                msg = "BOLA EM JOGO"
                color = (200, 240, 200)
        elif game_state.status == MatchStatus.GAME_OVER:
            msg = "FIM DE JOGO! FECHE A MAO P/ REINICIAR"
            color = (0, 230, 255)
        else:
            msg = ""
            color = (200, 200, 200)

        if msg:
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            pill_w = max(240, tw + 45)
            pill_x, pill_y = 20, 75
            pill_h = 30

            overlay = frame.copy()
            cv2.rectangle(overlay, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (16, 20, 30), -1)
            cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
            cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), color, 1)
            cv2.circle(frame, (pill_x + 13, pill_y + int(pill_h / 2)), 4, color, -1)
            cv2.putText(
                frame,
                msg,
                (pill_x + 26, pill_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                color,
                1,
            )

    def _render_hud(self, frame: np.ndarray, game_state: GameStateMachine):
        """Broadcast-style TV scorebug (clean floating scoreboard without obscuring table)."""
        w = self.width
        cx = int(w / 2)
        sb_w, sb_h = 420, 44
        sb_x = cx - int(sb_w / 2)
        sb_y = 14

        # 1. Subtle drop shadow
        shadow = frame.copy()
        cv2.rectangle(shadow, (sb_x + 2, sb_y + 2), (sb_x + sb_w + 2, sb_y + sb_h + 2), (10, 10, 15), -1)
        cv2.addWeighted(shadow, 0.40, frame, 0.60, 0, frame)

        # 2. Main glass container
        overlay = frame.copy()
        cv2.rectangle(overlay, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h), (16, 20, 28), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (sb_x, sb_y), (sb_x + sb_w, sb_y + sb_h), (60, 75, 95), 1)

        # 3. Left Pod: Player ("VOCE")
        p_pod_w = 160
        # Cyan accent stripe
        cv2.rectangle(frame, (sb_x, sb_y), (sb_x + 4, sb_y + sb_h), (255, 215, 0), -1)
        # Player name
        cv2.putText(frame, "VOCE", (sb_x + 14, sb_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.54, (255, 255, 255), 1)
        # Serve indicator
        if game_state.server == "PLAYER":
            cv2.circle(frame, (sb_x + 68, sb_y + 23), 5, (0, 165, 255), -1)
            cv2.circle(frame, (sb_x + 68, sb_y + 23), 7, (255, 255, 255), 1)

        # Player Digital Score Box
        cv2.rectangle(frame, (sb_x + p_pod_w - 42, sb_y + 4), (sb_x + p_pod_w - 4, sb_y + sb_h - 4), (30, 42, 58), -1)
        cv2.rectangle(frame, (sb_x + p_pod_w - 42, sb_y + 4), (sb_x + p_pod_w - 4, sb_y + sb_h - 4), (0, 200, 255), 1)
        p_score_str = str(game_state.player_score)
        p_score_x = sb_x + p_pod_w - 30 if len(p_score_str) == 1 else sb_x + p_pod_w - 36
        cv2.putText(frame, p_score_str, (p_score_x, sb_y + 31), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 235, 255), 2)

        # 4. Center Pod: Rally Counter (Width 100)
        c_pod_x = sb_x + p_pod_w
        cv2.line(frame, (c_pod_x, sb_y + 6), (c_pod_x, sb_y + sb_h - 6), (45, 55, 70), 1)
        cv2.putText(frame, "RALLY", (c_pod_x + 28, sb_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 215, 255), 1)
        rally_str = str(game_state.current_rally_count)
        r_text_x = c_pod_x + 44 if len(rally_str) == 1 else c_pod_x + 38
        cv2.putText(frame, rally_str, (r_text_x, sb_y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1)
        c_pod_end = c_pod_x + 100
        cv2.line(frame, (c_pod_end, sb_y + 6), (c_pod_end, sb_y + sb_h - 6), (45, 55, 70), 1)

        # 5. Right Pod: Roberval
        r_pod_x = c_pod_end
        # Roberval Digital Score Box
        cv2.rectangle(frame, (r_pod_x + 4, sb_y + 4), (r_pod_x + 42, sb_y + sb_h - 4), (50, 28, 34), -1)
        cv2.rectangle(frame, (r_pod_x + 4, sb_y + 4), (r_pod_x + 42, sb_y + sb_h - 4), (255, 90, 100), 1)
        r_score_str = str(game_state.ai_score)
        r_score_x = r_pod_x + 16 if len(r_score_str) == 1 else r_pod_x + 10
        cv2.putText(frame, r_score_str, (r_score_x, sb_y + 31), cv2.FONT_HERSHEY_DUPLEX, 0.75, (120, 140, 255), 2)

        # Serve indicator for Roberval
        if game_state.server in ("AI", "ROBERVAL"):
            cv2.circle(frame, (r_pod_x + 54, sb_y + 23), 5, (0, 165, 255), -1)
            cv2.circle(frame, (r_pod_x + 54, sb_y + 23), 7, (255, 255, 255), 1)

        # Roberval name
        r_name_x = r_pod_x + 68 if game_state.server in ("AI", "ROBERVAL") else r_pod_x + 52
        cv2.putText(frame, "ROBERVAL", (r_name_x, sb_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1)

        # Red accent stripe on right edge
        cv2.rectangle(frame, (sb_x + sb_w - 4, sb_y), (sb_x + sb_w, sb_y + sb_h), (60, 60, 235), -1)
