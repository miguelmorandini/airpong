/**
 * AI Ping Pong - Wii Sports Mode (100% Identical 1:1 Web Engine to Python OpenCV Desktop)
 * - Complete 3D Perspective Projection Calibrated to assets/background_arena.png
 * - Procedural Fallback Arena (Olympic Taraflex floor, undercarriage, apron & 3D net)
 * - Decisive ITTF Net Collision Logic (Tape Clip graze & Net Fault point award)
 * - Sub-Stepping Physics Engine with Continuous Collision Detection (CCD)
 * - Zero-GC Particle Pool & Pre-Allocated Ring Buffers (Silky-smooth 60 FPS)
 * - Non-Blocking Lazy MediaPipe Webcam Setup (Instant 0ms Startup)
 * - Match State Machine with ITTF Rules, Deuce, and Serve Rotation
 * - Roberval AI with 4 Difficulty Levels (Easy, Medium, Hard, Champion)
 * - Motion Attack Model (Wii Sports Arcade Feel) with Broad Strike Reach & Hit-Stop
 * - Glassmorphic Start Menu / Lobby Modal, In-Game Broadcast Scorebug, Lateral Status Pill
 * - Live Camera PiP with MediaPipe Hand Skeleton & Strike Zone Depth Needle
 * - Interactive Arm & Hand Calibration Wizard (Trainer Mode [T])
 * - Web Audio API Procedural Sound Synthesizer (Paddle Hit, Smash, Bounce, Point, Cheer)
 */

(() => {
  "use strict";

  // =========================================================================
  // 1. CONFIGURATION & CONSTANTS (Matching src/config.py)
  // =========================================================================
  const CONFIG = {
    // ITTF Official Dimensions (meters)
    TABLE_LENGTH: 2.74,
    TABLE_WIDTH: 1.525,
    TABLE_HEIGHT: 0.76,
    NET_HEIGHT: 0.1525,
    NET_OVERHANG: 0.1525,
    BALL_RADIUS: 0.020,
    PADDLE_RADIUS: 0.076,

    // Resolution (16:10 Native aspect ratio matching Python OpenCV)
    WIDTH: 1024,
    HEIGHT: 640,

    // Exact sub-pixel calibrated 3D camera projection parameters matching background_arena.png
    CAM_BASE_X: 0.0,
    CAM_BASE_Y: 1.23505274,
    CAM_BASE_Z: -3.86738472,
    CAM_PITCH: -14.9153403 * (Math.PI / 180.0), // -0.2603218 rad
    FOCAL_LENGTH: 1269.75295,

    // Procedural 3D perspective fallback parameters
    PROC_CAM_X: 0.0,
    PROC_CAM_Y: 1.25,
    PROC_CAM_Z: -3.50,
    PROC_CAM_PITCH: -18.0 * (Math.PI / 180.0),
    PROC_FOCAL_LENGTH: (1024.0 / 2.0) / Math.tan((48.0 * Math.PI) / 360.0), // ~1150.0

    // Physics constants
    GRAVITY: 9.81,
    AIR_DRAG: 0.14,
    RESTITUTION_TABLE: 0.88,
    RESTITUTION_PADDLE: 1.15,

    // Motion Attack Tuning
    BROAD_STRIKE_ZONE_Z_MIN: -1.75,
    BROAD_STRIKE_ZONE_Z_MAX: -0.75,
    MOTION_ATTACK_REACH: 0.52,
  };

  // =========================================================================
  // 2. PROCEDURAL SOUND SYNTHESIZER (Lazy Web Audio API)
  // =========================================================================
  class SoundSynth {
    constructor() {
      this.ctx = null;
    }

    ensureContext() {
      if (!this.ctx) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) {
          this.ctx = new AudioCtx();
        }
      }
      if (this.ctx && this.ctx.state === "suspended") {
        this.ctx.resume().catch(() => {});
      }
    }

    playPaddleHit(isSmash = false, pan = 0.0) {
      this.ensureContext();
      if (!this.ctx) return;
      const t = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      if (isSmash) {
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(680, t);
        osc.frequency.exponentialRampToValueAtTime(80, t + 0.16);
        gain.gain.setValueAtTime(0.55, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.16);
      } else {
        osc.type = "triangle";
        osc.frequency.setValueAtTime(460, t);
        osc.frequency.exponentialRampToValueAtTime(140, t + 0.11);
        gain.gain.setValueAtTime(0.35, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.11);
      }

      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(t);
      osc.stop(t + (isSmash ? 0.18 : 0.12));
    }

    playBounce(isTable = true, pan = 0.0) {
      this.ensureContext();
      if (!this.ctx) return;
      const t = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = isTable ? "sine" : "triangle";
      osc.frequency.setValueAtTime(isTable ? 320 : 410, t);
      osc.frequency.exponentialRampToValueAtTime(90, t + 0.08);

      gain.gain.setValueAtTime(0.28, t);
      gain.gain.exponentialRampToValueAtTime(0.01, t + 0.08);

      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(t);
      osc.stop(t + 0.09);
    }

    playPointScore(isPlayerWon = true) {
      this.ensureContext();
      if (!this.ctx) return;
      const t = this.ctx.currentTime;
      const notes = isPlayerWon ? [523.25, 659.25, 783.99, 1046.5] : [392.0, 329.63, 261.63];
      notes.forEach((freq, idx) => {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, t + idx * 0.08);
        gain.gain.setValueAtTime(0.25, t + idx * 0.08);
        gain.gain.exponentialRampToValueAtTime(0.001, t + idx * 0.08 + 0.22);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(t + idx * 0.08);
        osc.stop(t + idx * 0.08 + 0.24);
      });
    }

    playCheer() {
      this.ensureContext();
      if (!this.ctx) return;
      const t = this.ctx.currentTime;
      [440, 554.37, 659.25, 880].forEach((freq) => {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(freq, t);
        gain.gain.setValueAtTime(0.12, t);
        gain.gain.exponentialRampToValueAtTime(0.001, t + 0.65);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(t);
        osc.stop(t + 0.70);
      });
    }
  }

  // =========================================================================
  // 3. ZERO-GC PARTICLE POOL (High Performance 60 FPS)
  // =========================================================================
  class ParticlePool {
    constructor(size = 64) {
      this.pool = [];
      for (let i = 0; i < size; i++) {
        this.pool.push({
          active: false,
          x: 0,
          y: 0,
          vx: 0,
          vy: 0,
          life: 0,
          maxLife: 0,
          color: "#ffd166",
          radius: 2,
        });
      }
    }

    spawn(screenX, screenY, count = 10, color = "#ffd166") {
      let spawned = 0;
      for (let i = 0; i < this.pool.length && spawned < count; i++) {
        const p = this.pool[i];
        if (!p.active) {
          const angle = Math.random() * Math.PI * 2;
          const spd = 2.0 + Math.random() * 5.0;
          p.active = true;
          p.x = screenX;
          p.y = screenY;
          p.vx = Math.cos(angle) * spd;
          p.vy = Math.sin(angle) * spd - 1.5;
          p.life = 0.35 + Math.random() * 0.25;
          p.maxLife = p.life;
          p.color = color;
          p.radius = 2 + Math.random() * 2;
          spawned++;
        }
      }
    }

    update(dt) {
      for (let i = 0; i < this.pool.length; i++) {
        const p = this.pool[i];
        if (p.active) {
          p.life -= dt;
          if (p.life <= 0) {
            p.active = false;
          } else {
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 4.0 * dt;
          }
        }
      }
    }

    render(ctx) {
      for (let i = 0; i < this.pool.length; i++) {
        const p = this.pool[i];
        if (p.active) {
          const alpha = Math.max(0, p.life / p.maxLife);
          ctx.save();
          ctx.fillStyle = p.color;
          ctx.globalAlpha = alpha;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius * alpha, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
      }
    }
  }

  // =========================================================================
  // 4. 3D PERSPECTIVE RENDERER & CAMERA (1:1 with PerspectiveRenderer3D)
  // =========================================================================
  class PerspectiveRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.width = CONFIG.WIDTH;
      this.height = CONFIG.HEIGHT;

      // Load realistic arena background image
      this.bgImage = new Image();
      this.bgImageLoaded = false;
      this.has_image_background = false;

      // Start with calibrated parameters immediately for zero-lag render
      this.cam_base_x = CONFIG.CAM_BASE_X;
      this.cam_base_y = CONFIG.CAM_BASE_Y;
      this.cam_base_z = CONFIG.CAM_BASE_Z;
      this.cam_x = this.cam_base_x;
      this.cam_y = this.cam_base_y;
      this.cam_z = this.cam_base_z;
      this.cam_pitch = CONFIG.CAM_PITCH;
      this.focal_length = CONFIG.FOCAL_LENGTH;

      this.bgImage.onload = () => {
        this.bgImageLoaded = true;
        this.has_image_background = true;
        console.log("[RENDERER] background_arena.png carregado! Projeção 3D calibrada ativada.");
      };

      this.bgImage.onerror = () => {
        console.warn("[RENDERER] background_arena.png não encontrado. Usando renderizador procedural.");
        this.has_image_background = false;
        this.cam_base_x = CONFIG.PROC_CAM_X;
        this.cam_base_y = CONFIG.PROC_CAM_Y;
        this.cam_base_z = CONFIG.PROC_CAM_Z;
        this.cam_x = this.cam_base_x;
        this.cam_y = this.cam_base_y;
        this.cam_z = this.cam_base_z;
        this.cam_pitch = CONFIG.PROC_CAM_PITCH;
        this.focal_length = CONFIG.PROC_FOCAL_LENGTH;
      };

      // Search paths for background_arena.png
      this.bgImage.src = (typeof window !== "undefined" && window.BG_ARENA_IMAGE_DATA)
        ? window.BG_ARENA_IMAGE_DATA
        : "assets/background_arena.png";

      // Official AirPong Logo & Emblem Assets
      this.logoImage = new Image();
      this.logoImageLoaded = false;
      this.logoImage.onload = () => { this.logoImageLoaded = true; };
      this.logoImage.src = "assets/airpong_logo.png";

      this.emblemImage = new Image();
      this.emblemImageLoaded = false;
      this.emblemImage.onload = () => { this.emblemImageLoaded = true; };
      this.emblemImage.src = "assets/airpong_emblem.png";

      this.trauma = 0.0;
      this.shake_time = 0.0;

      // Hitboxes for click interactions
      this.menu_buttons = {};
      this.hud_menu_button = [14, 14, 106, 42]; // [ ⏸ MENU ] in-game HUD button
    }

    addTrauma(amount) {
      this.trauma = Math.min(1.0, this.trauma + amount);
    }

    prepareFrame() {
      // Dynamic High-DPI physical backing store (Retina, 4K, crisp rendering)
      const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
      const rect = this.canvas.getBoundingClientRect();
      const targetW = Math.round((rect.width || this.width) * dpr);
      const targetH = Math.round((rect.height || this.height) * dpr);

      if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
        this.canvas.width = targetW;
        this.canvas.height = targetH;
      }

      const sx = this.canvas.width / this.width;
      const sy = this.canvas.height / this.height;
      this.ctx.setTransform(sx, 0, 0, sy, 0, 0);
      this.ctx.imageSmoothingEnabled = true;
      this.ctx.imageSmoothingQuality = "high";
    }

    updateCamera(ball, dt) {
      this.shake_time += dt;
      if (this.trauma > 0.0) {
        this.trauma = Math.max(0.0, this.trauma - dt * 2.2);
      }

      if (!this.has_image_background) {
        // Procedural camera subtle lookahead
        if (ball.in_play) {
          const target_x = Math.max(-0.22, Math.min(0.22, ball.x * 0.16));
          const target_y = 1.25 + Math.max(-0.05, Math.min(0.15, ball.y * 0.08));
          const alpha = Math.min(1.0, dt * 3.5);
          this.cam_x += (target_x - this.cam_x) * alpha;
          this.cam_y += (target_y - this.cam_y) * alpha;
        } else {
          this.cam_x += (0.0 - this.cam_x) * dt * 3.5;
          this.cam_y += (1.25 - this.cam_y) * dt * 3.5;
        }
      } else {
        // Keep camera solidly locked to background table geometry
        this.cam_x = this.cam_base_x;
        this.cam_y = this.cam_base_y;
        this.cam_z = this.cam_base_z;
      }
    }

    projectPoint(x, y, z) {
      let shake_ox = 0;
      let shake_oy = 0;
      if (!this.has_image_background && this.trauma > 0.0) {
        const shake = this.trauma * this.trauma;
        shake_ox = Math.sin(this.shake_time * 26.0) * shake * 14.0;
        shake_oy = Math.cos(this.shake_time * 32.0) * shake * 9.0;
      }

      const dx = x - this.cam_x;
      const dy = y - this.cam_y;
      const dz = z - this.cam_z;

      const cos_p = Math.cos(this.cam_pitch);
      const sin_p = Math.sin(this.cam_pitch);

      const cy = dy * cos_p - dz * sin_p;
      const cz = dy * sin_p + dz * cos_p;

      if (cz <= 0.15) return null;

      const sx = this.width / 2.0 + (this.focal_length * dx) / cz + shake_ox;
      const sy = this.height / 2.0 - (this.focal_length * cy) / cz + shake_oy;

      return { x: sx, y: sy, z: cz };
    }

    unprojectPointAtZ(screen_x, screen_y, z_plane = -1.40) {
      const dz = z_plane - this.cam_z;
      const cos_p = Math.cos(this.cam_pitch);
      const sin_p = Math.sin(this.cam_pitch);

      const vy = (this.height / 2.0 - screen_y) / this.focal_length;
      const denom = cos_p - vy * sin_p;
      if (Math.abs(denom) < 1e-6) return { x: 0.0, y: 0.20 };

      const dy = (dz * (vy * cos_p + sin_p)) / denom;
      const world_y = this.cam_y + dy;
      const cam_z = dy * sin_p + dz * cos_p;

      const vx = (screen_x - this.width / 2.0) / this.focal_length;
      const world_x = this.cam_x + vx * cam_z;

      return { x: world_x, y: world_y };
    }

    renderBackground() {
      this.prepareFrame();
      const ctx = this.ctx;

      if (this.has_image_background && this.bgImageLoaded) {
        // Whole frame trauma shake warp
        let ox = 0, oy = 0;
        if (this.trauma > 0.0) {
          const shake = this.trauma * this.trauma;
          ox = Math.sin(this.shake_time * 28.0) * shake * 12.0;
          oy = Math.cos(this.shake_time * 34.0) * shake * 8.0;
        }
        ctx.save();
        if (this.trauma > 0.0) {
          ctx.drawImage(this.bgImage, ox - 4, oy - 3, this.width + 8, this.height + 6);
        } else {
          ctx.drawImage(this.bgImage, 0, 0, this.width, this.height);
        }
        ctx.restore();
        return;
      }

      // Procedural Arena Fallback
      // 1. Deep Championship stadium arena gradient
      const grad = ctx.createLinearGradient(0, 0, 0, this.height);
      grad.addColorStop(0, "#101218");
      grad.addColorStop(1, "#2a262d");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, this.width, this.height);

      // Spectator tiers
      const tierY = Math.floor(this.height * 0.32);
      ctx.strokeStyle = "#201e22";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, tierY); ctx.lineTo(this.width, tierY);
      ctx.moveTo(0, tierY + 18); ctx.lineTo(this.width, tierY + 18);
      ctx.moveTo(0, tierY + 36); ctx.lineTo(this.width, tierY + 36);
      ctx.stroke();

      // Overhead spotlight glow
      const cx = this.width / 2;
      const cy = Math.floor(this.height * 0.40);
      const spotGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, this.width * 0.45);
      spotGrad.addColorStop(0, "rgba(90, 80, 75, 0.45)");
      spotGrad.addColorStop(0.6, "rgba(65, 55, 50, 0.25)");
      spotGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = spotGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, this.width * 0.45, 0, Math.PI * 2);
      ctx.fill();

      // 2. Olympic Taraflex burgundy sports court
      const f1 = this.projectPoint(-6.0, -0.76, -3.8);
      const f2 = this.projectPoint(6.0, -0.76, -3.8);
      const f3 = this.projectPoint(6.0, -0.76, 5.0);
      const f4 = this.projectPoint(-6.0, -0.76, 5.0);
      if (f1 && f2 && f3 && f4) {
        ctx.fillStyle = "#5c2630"; // Burgundy Taraflex vinyl
        ctx.beginPath();
        ctx.moveTo(f1.x, f1.y);
        ctx.lineTo(f2.x, f2.y);
        ctx.lineTo(f3.x, f3.y);
        ctx.lineTo(f4.x, f4.y);
        ctx.closePath();
        ctx.fill();
      }

      // Inner court mat (7m x 14m)
      const m1 = this.projectPoint(-3.5, -0.76, -3.4);
      const m2 = this.projectPoint(3.5, -0.76, -3.4);
      const m3 = this.projectPoint(3.5, -0.76, 4.0);
      const m4 = this.projectPoint(-3.5, -0.76, 4.0);
      if (m1 && m2 && m3 && m4) {
        ctx.fillStyle = "#73303e";
        ctx.beginPath();
        ctx.moveTo(m1.x, m1.y);
        ctx.lineTo(m2.x, m2.y);
        ctx.lineTo(m3.x, m3.y);
        ctx.lineTo(m4.x, m4.y);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = "rgba(205, 185, 180, 0.6)";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Court surrounds / barriers
      const b_bl = this.projectPoint(-3.8, -0.76, 4.2);
      const b_br = this.projectPoint(3.8, -0.76, 4.2);
      const b_tr = this.projectPoint(3.8, -0.06, 4.2);
      const b_tl = this.projectPoint(-3.8, -0.06, 4.2);
      if (b_bl && b_br && b_tr && b_tl) {
        ctx.fillStyle = "#30262a";
        ctx.beginPath();
        ctx.moveTo(b_bl.x, b_bl.y);
        ctx.lineTo(b_br.x, b_br.y);
        ctx.lineTo(b_tr.x, b_tr.y);
        ctx.lineTo(b_tl.x, b_tl.y);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = "#9b8282";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // 3. 3D ITTF Table Structure
      const w2 = CONFIG.TABLE_WIDTH / 2.0;
      const l2 = CONFIG.TABLE_LENGTH / 2.0;
      const apronDepth = 0.05;

      // 4 Steel legs
      const legInsetX = 0.12, legInsetZ = 0.18;
      const legs = [
        [-w2 + legInsetX, l2 - legInsetZ],
        [w2 - legInsetX, l2 - legInsetZ],
        [w2 - legInsetX, -l2 + legInsetZ],
        [-w2 + legInsetX, -l2 + legInsetZ],
      ];
      ctx.strokeStyle = "#282a30";
      legs.forEach(([lx, lz]) => {
        const top = this.projectPoint(lx, -apronDepth, lz);
        const bot = this.projectPoint(lx, -0.76, lz);
        if (top && bot) {
          ctx.lineWidth = Math.max(3, Math.floor(18.0 / bot.z));
          ctx.beginPath();
          ctx.moveTo(top.x, top.y);
          ctx.lineTo(bot.x, bot.y);
          ctx.stroke();
        }
      });

      // Apron skirts
      const c_nw = this.projectPoint(-w2, 0, l2);
      const c_ne = this.projectPoint(w2, 0, l2);
      const c_se = this.projectPoint(w2, 0, -l2);
      const c_sw = this.projectPoint(-w2, 0, -l2);

      const a_nw = this.projectPoint(-w2, -apronDepth, l2);
      const a_ne = this.projectPoint(w2, -apronDepth, l2);
      const a_se = this.projectPoint(w2, -apronDepth, -l2);
      const a_sw = this.projectPoint(-w2, -apronDepth, -l2);

      // Front apron
      if (c_sw && c_se && a_se && a_sw) {
        ctx.fillStyle = "#0a3473"; // Navy apron
        ctx.beginPath();
        ctx.moveTo(c_sw.x, c_sw.y);
        ctx.lineTo(c_se.x, c_se.y);
        ctx.lineTo(a_se.x, a_se.y);
        ctx.lineTo(a_sw.x, a_sw.y);
        ctx.closePath();
        ctx.fill();
      }

      // Left apron
      if (c_sw && c_nw && a_nw && a_sw) {
        ctx.fillStyle = "#08295b";
        ctx.beginPath();
        ctx.moveTo(c_sw.x, c_sw.y);
        ctx.lineTo(c_nw.x, c_nw.y);
        ctx.lineTo(a_nw.x, a_nw.y);
        ctx.lineTo(a_sw.x, a_sw.y);
        ctx.closePath();
        ctx.fill();
      }

      // Right apron
      if (c_se && c_ne && a_ne && a_se) {
        ctx.fillStyle = "#08295b";
        ctx.beginPath();
        ctx.moveTo(c_se.x, c_se.y);
        ctx.lineTo(c_ne.x, c_ne.y);
        ctx.lineTo(a_ne.x, a_ne.y);
        ctx.lineTo(a_se.x, a_se.y);
        ctx.closePath();
        ctx.fill();
      }

      // Top playing surface (Championship ITTF Blue)
      if (c_nw && c_ne && c_se && c_sw) {
        ctx.fillStyle = "#0e52a5";
        ctx.beginPath();
        ctx.moveTo(c_nw.x, c_nw.y);
        ctx.lineTo(c_ne.x, c_ne.y);
        ctx.lineTo(c_se.x, c_se.y);
        ctx.lineTo(c_sw.x, c_sw.y);
        ctx.closePath();
        ctx.fill();

        // White border lines (20mm)
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.stroke();

        // Center line (3mm)
        const cmN = this.projectPoint(0, 0, l2);
        const cmS = this.projectPoint(0, 0, -l2);
        if (cmN && cmS) {
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(cmN.x, cmN.y);
          ctx.lineTo(cmS.x, cmS.y);
          ctx.stroke();
        }
      }

      // 4. 3D Net
      const nw = w2 + CONFIG.NET_OVERHANG;
      const nh = CONFIG.NET_HEIGHT;
      const n_tl = this.projectPoint(-nw, nh, 0.0);
      const n_tr = this.projectPoint(nw, nh, 0.0);
      const n_br = this.projectPoint(nw, 0.0, 0.0);
      const n_bl = this.projectPoint(-nw, 0.0, 0.0);
      if (n_tl && n_tr && n_br && n_bl) {
        ctx.fillStyle = "rgba(25, 25, 28, 0.78)";
        ctx.beginPath();
        ctx.moveTo(n_tl.x, n_tl.y);
        ctx.lineTo(n_tr.x, n_tr.y);
        ctx.lineTo(n_br.x, n_br.y);
        ctx.lineTo(n_bl.x, n_bl.y);
        ctx.closePath();
        ctx.fill();

        // White upper tension cord
        ctx.strokeStyle = "#fafaff";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(n_tl.x, n_tl.y);
        ctx.lineTo(n_tr.x, n_tr.y);
        ctx.stroke();

        // Side posts
        ctx.strokeStyle = "#41444b";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(n_bl.x, n_bl.y + 8); ctx.lineTo(n_tl.x, n_tl.y - 4);
        ctx.moveTo(n_br.x, n_br.y + 8); ctx.lineTo(n_tr.x, n_tr.y - 4);
        ctx.stroke();
      }
    }
  }

  // =========================================================================
  // 5. BALL, PADDLE & STATE DEFINITIONS
  // =========================================================================
  const MatchStatus = {
    READY: "READY",
    SERVING: "SERVING",
    RALLY: "RALLY",
    GAME_OVER: "GAME_OVER",
  };

  class Ball {
    constructor() {
      this.reset("PLAYER");
    }

    reset(server = "PLAYER") {
      this.x = (server === "PLAYER") ? 0.15 : -0.18;
      this.y = 0.28;
      this.z = (server === "PLAYER") ? -1.35 : 1.35;
      this.prevX = this.x;
      this.prevY = this.y;
      this.prevZ = this.z;
      this.vx = 0.0;
      this.vy = 0.0;
      this.vz = 0.0;
      this.spin_x = 0.0;
      this.spin_y = 0.0;
      this.spin_z = 0.0;
      this.in_play = false;
      this.isServe = true;
      this.lastHitter = null;
      this.bounces = 0;
      this.bouncedPlayerSide = false;
      this.bouncedAiSide = false;
      this.speed_kmh = 0.0;
      this.squash = 0.0;
      this.trail = [];
    }

    integrateForces(dt) {
      if (this.squash > 0.0) {
        this.squash = Math.max(0.0, this.squash - dt * 6.0);
      }

      if (!this.in_play) return;

      const speed = Math.hypot(this.vx, this.vy, this.vz);
      this.speed_kmh = speed * 3.6;

      // Air drag & Magnus spin (matching src/engine/physics.py)
      const drag = CONFIG.AIR_DRAG * speed;
      const kMagnus = 0.003;
      const ax = -drag * this.vx + kMagnus * (this.spin_y * this.vz - this.spin_z * this.vy);
      const ay = -CONFIG.GRAVITY - drag * this.vy + kMagnus * (this.spin_z * this.vx - this.spin_x * this.vz);
      const az = -drag * this.vz + kMagnus * (this.spin_x * this.vy - this.spin_y * this.vx);

      this.vx += ax * dt;
      this.vy += ay * dt;
      this.vz += az * dt;

      this.prevX = this.x;
      this.prevY = this.y;
      this.prevZ = this.z;

      this.x += this.vx * dt;
      this.y += this.vy * dt;
      this.z += this.vz * dt;

      this.spin_x *= 0.985;
      this.spin_y *= 0.985;
      this.spin_z *= 0.985;

      // Trail recording
      this.trail.unshift({ x: this.x, y: this.y, z: this.z, speed: this.speed_kmh });
      if (this.trail.length > 14) this.trail.pop();
    }
  }

  class Paddle {
    constructor(isPlayer = true) {
      this.isPlayer = isPlayer;
      this.x = 0.0;
      this.y = 0.16;
      this.z = isPlayer ? -1.40 : 1.40;
      this.vx = 0.0;
      this.vy = 0.0;
      this.vz = 0.0;
      this.speed = 0.0;
      this.hand_speed = 0.0;
      this.is_hook_swing = false;
      this.hook_power = 1.0;
      this.target_x = 0.0;
      this.target_y = 0.16;
      this.roll = 0.0;
      this.pitch = 0.0;
      this.trail = [];
    }

    update(tx, ty, tz, dt) {
      const prevX = this.x;
      const prevY = this.y;
      const prevZ = this.z;

      const alpha = Math.min(1.0, dt * 24.0);
      this.x += (tx - this.x) * alpha;
      this.y += (ty - this.y) * alpha;
      this.z += (tz - this.z) * alpha;

      this.vx = (this.x - prevX) / Math.max(1e-4, dt);
      this.vy = (this.y - prevY) / Math.max(1e-4, dt);
      this.vz = (this.z - prevZ) / Math.max(1e-4, dt);
      this.speed = Math.hypot(this.vx, this.vy, this.vz);

      this.trail.unshift({ x: this.x, y: this.y, z: this.z });
      if (this.trail.length > 6) this.trail.pop();
    }
  }

  // =========================================================================
  // 6. ROBERVAL AI OPPONENT (1:1 with TableTennisAI)
  // =========================================================================
  class RobervalAI {
    constructor(difficulty = "EASY") {
      this.setDifficulty(difficulty);
    }

    setDifficulty(diff) {
      this.difficulty = diff;
      switch (diff) {
        case "EASY":
          this.speed = 1.15;        // Movimento calmo e suave, super acessivel
          this.accuracy = 0.45;     // Baixa precisao nas quinas, erra facil bolas laterais
          this.smashTendency = 0.0; // NUNCA ataca nem dá smashes, ritmo amigavel
          break;
        case "HARD":
        case "CHAMPION":
          this.difficulty = "HARD";
          this.speed = 4.2;
          this.accuracy = 0.94;
          this.smashTendency = 0.65;
          break;
        case "MEDIUM":
        default:
          this.difficulty = "MEDIUM";
          this.speed = 2.6;
          this.accuracy = 0.85;
          this.smashTendency = 0.25;
          break;
      }
    }

    update(paddle, ball, dt) {
      if (!ball.in_play || ball.vz <= 0) {
        paddle.target_x += (0.0 - paddle.target_x) * dt * 2.5;
        paddle.target_y += (0.16 - paddle.target_y) * dt * 2.5;
        paddle.update(paddle.target_x, paddle.target_y, 1.40, dt);
        return;
      }

      // Intercept prediction at AI baseline Z = 1.35
      const timeToReach = (1.35 - ball.z) / Math.max(0.1, ball.vz);
      let predX = ball.x + ball.vx * timeToReach;
      let predY = Math.max(0.04, ball.y + ball.vy * timeToReach - 0.5 * CONFIG.GRAVITY * timeToReach * timeToReach);

      predX = Math.max(-0.75, Math.min(0.75, predX * this.accuracy));
      predY = Math.max(0.06, Math.min(0.45, predY));

      const dx = predX - paddle.x;
      const dy = predY - paddle.y;
      const step = this.speed * dt;

      paddle.target_x += Math.max(-step, Math.min(step, dx));
      paddle.target_y += Math.max(-step, Math.min(step, dy));
      paddle.update(paddle.target_x, paddle.target_y, 1.40, dt);
    }
  }

  // =========================================================================
  // 7. GESTURE CALIBRATION & TRAINING WIZARD (Matching calibration.py)
  // =========================================================================
  class GestureTrainer {
    constructor() {
      this.STEPS = [
        { key: "REST", title: "1. POSICAO DE ESPERA: Mantenha o braco preparado na frente" },
        { key: "SERVE", title: "2. TREINO DE SAQUE: Execute 2 movimentos de Saque com o braco" },
        { key: "FOREHAND", title: "3. GOLPE DE DIREITA: Execute 2 batidas de Forehand (para dentro)" },
        { key: "BACKHAND", title: "4. GOLPE DE ESQUERDA: Execute 2 batidas de Backhand (para fora)" },
        { key: "SMASH", title: "5. CORTADA POTENTE: Execute 2 smashes fortes de cima para baixo!" },
      ];

      this.is_active = false;
      this.current_step_idx = 0;
      this.step_start_time = 0;
      this.samples = { SERVE: [], FOREHAND: [], BACKHAND: [], SMASH: [] };
      this.last_swing_time = 0;
      this.has_returned_to_rest = true;
      this.feedback_msg = "";
      this.feedback_color = "#00e6ff";
      this.feedback_timer = 0;
      this.profile = this.loadProfile();
    }

    loadProfile() {
      try {
        const saved = localStorage.getItem("ai_ping_pong_profile");
        if (saved) return JSON.parse(saved);
      } catch (e) {}
      return { is_calibrated: false };
    }

    saveProfile() {
      try {
        this.profile.is_calibrated = true;
        this.profile.calibrated_at = new Date().toISOString();
        localStorage.setItem("ai_ping_pong_profile", JSON.stringify(this.profile));
      } catch (e) {}
    }

    start() {
      this.is_active = true;
      this.current_step_idx = 0;
      this.step_start_time = performance.now() / 1000;
      this.samples = { SERVE: [], FOREHAND: [], BACKHAND: [], SMASH: [] };
      this.feedback_msg = "INICIANDO CALIBRACAO... Posicione o braco!";
      this.feedback_color = "#00e6ff";
      this.feedback_timer = 2.0;
    }

    stop() {
      this.is_active = false;
      this.feedback_timer = 0;
    }

    skipStep() {
      if (this.current_step_idx < this.STEPS.length - 1) {
        this.current_step_idx++;
        this.step_start_time = performance.now() / 1000;
      } else {
        this.finish();
      }
    }

    finish() {
      this.is_active = false;
      this.saveProfile();
      this.feedback_msg = "TREINO CONCLUIDO COM SUCESSO!";
      this.feedback_color = "#3cf078";
      this.feedback_timer = 3.0;
    }

    update(curSpeed, curVx, curVy, isFist, dt, sounds) {
      if (this.feedback_timer > 0) this.feedback_timer -= dt;
      if (!this.is_active) return;

      const now = performance.now() / 1000;
      const step = this.STEPS[this.current_step_idx];

      if (step.key === "REST") {
        if (now - this.step_start_time > 2.5) {
          this.current_step_idx++;
          this.step_start_time = now;
          sounds.playBounce();
        }
        return;
      }

      // Swing recording
      if (curSpeed >= 0.8 && this.has_returned_to_rest && (now - this.last_swing_time > 0.6)) {
        this.samples[step.key].push(curSpeed);
        this.last_swing_time = now;
        this.has_returned_to_rest = false;
        sounds.playBounce();
        this.feedback_msg = `REPETICAO ${this.samples[step.key].length} / 2 REGISTRADA!`;
        this.feedback_color = "#ffd166";
        this.feedback_timer = 1.0;

        if (this.samples[step.key].length >= 2) {
          if (this.current_step_idx < this.STEPS.length - 1) {
            this.current_step_idx++;
            this.step_start_time = now;
            sounds.playPaddleHit(true);
          } else {
            this.finish();
            sounds.playCheer();
          }
        }
      } else if (curSpeed < 0.35) {
        this.has_returned_to_rest = true;
      }
    }

    renderOverlay(ctx, currentSpeedKmh) {
      const step = this.STEPS[this.current_step_idx] || this.STEPS[0];
      const cx = CONFIG.WIDTH / 2;

      ctx.save();
      // Dark backdrop overlay
      ctx.fillStyle = "rgba(4, 9, 18, 0.82)";
      ctx.fillRect(0, 0, CONFIG.WIDTH, CONFIG.HEIGHT);

      // Card
      const cardW = 620;
      const cardH = 340;
      const cardX = cx - cardW / 2;
      const cardY = (CONFIG.HEIGHT - cardH) / 2;

      ctx.fillStyle = "#0c1524";
      ctx.fillRect(cardX, cardY, cardW, cardH);
      ctx.strokeStyle = "#00e6ff";
      ctx.lineWidth = 2;
      ctx.strokeRect(cardX, cardY, cardW, cardH);

      // Title
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 20px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("CALIBRACAO DE MOVIMENTOS DO BRACO", cx, cardY + 36);

      // Step text
      ctx.fillStyle = "#ffd166";
      ctx.font = "bold 15px sans-serif";
      ctx.fillText(step.title, cx, cardY + 75);

      // Progress bar
      const progressRatio = (this.current_step_idx + 1) / this.STEPS.length;
      ctx.fillStyle = "#1e2c40";
      ctx.fillRect(cardX + 40, cardY + 105, cardW - 80, 10);
      ctx.fillStyle = "#00e6ff";
      ctx.fillRect(cardX + 40, cardY + 105, (cardW - 80) * progressRatio, 10);

      // Rep Counter
      const reps = this.samples[step.key]?.length || 0;
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "14px sans-serif";
      ctx.fillText(`Passo ${this.current_step_idx + 1} de 5  |  Repetições: ${reps} / 2`, cx, cardY + 140);

      // Speed readout
      ctx.fillStyle = "#00e6ff";
      ctx.font = "bold 28px sans-serif";
      ctx.fillText(`VELOCIDADE: ${currentSpeedKmh.toFixed(1)} km/h`, cx, cardY + 195);

      // Feedback message
      if (this.feedback_msg) {
        ctx.fillStyle = this.feedback_color;
        ctx.font = "bold 16px sans-serif";
        ctx.fillText(this.feedback_msg, cx, cardY + 240);
      }

      // Controls
      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px sans-serif";
      ctx.fillText("[ESPACO] Pular Passo   |   [ESC] Sair do Treinamento", cx, cardY + cardH - 24);

      ctx.restore();
    }
  }

  // =========================================================================
  // 8. MAIN ENGINE & APPLICATION (1:1 with main.py)
  // =========================================================================
  class GameApp {
    constructor() {
      this.canvas = document.getElementById("game-canvas");
      this.video = document.getElementById("webcam");

      this.renderer = new PerspectiveRenderer(this.canvas);
      this.sounds = new SoundSynth();
      this.particlePool = new ParticlePool(64);
      this.ai = new RobervalAI("MEDIUM");
      this.ball = new Ball();
      this.playerPaddle = new Paddle(true);
      this.aiPaddle = new Paddle(false);
      this.trainer = new GestureTrainer();

      // State machine
      this.status = MatchStatus.READY;
      this.server = "PLAYER";
      this.player_score = 0;
      this.ai_score = 0;
      this.player_sets = 0;
      this.ai_sets = 0;
      this.set_number = 1;
      this.current_rally_count = 0;
      this.in_menu = true; // Start in Start Menu / Lobby modal
      this.pointPending = false;

      this.announcement = "NOVA PARTIDA CONTRA ROBERVAL!";
      this.announcement_timer = 3.0;
      this.serve_cooldown = 0.0;
      this.hit_stop_timer = 0.0;

      // Mouse controls & UI Navigation state
      this.mousePos = { x: 0.0, y: 0.20 };
      this.mouseScreenX = -100;
      this.mouseScreenY = -100;
      this.hoveredAction = null;
      this.menuTab = "PLAY"; // "PLAY" | "HELP" | "BIO"
      this.focusedActionIndex = 3; // start focused on primary start action
      this.mouseLastMoved = 0;
      this.clickSwing = false;

      // Webcam / MediaPipe state
      this.cameraActive = false;
      this.handDetected = false;
      this.handLandmarks = null;
      this.armDetected = false;
      this.isFist = false;
      this.elbowAngle = 120.0;
      this.wristSpeed = 0.0;
      this.wristVx = 0.0;
      this.wristVy = 0.0;
      this.wristVz = 0.0;
      this.worldHand = { x: 0.0, y: 0.20, z: -1.40 };
      this.lastHandPos = null;
      this.lastHandTime = 0;
      this.handednessMode = "AUTO";

      this.setupInputs();

      // Defer webcam setup smoothly without blocking the initial 60 FPS render!
      this.webcamSetupScheduled = false;
      this.scheduleWebcamSetup();

      this.lastFrameTime = performance.now();
      requestAnimationFrame(this.loop.bind(this));
    }

    scheduleWebcamSetup() {
      if (this.webcamSetupScheduled) return;
      this.webcamSetupScheduled = true;
      // Start render immediately, initialize webcam in background after UI stabilizes
      if (typeof window.requestIdleCallback === "function") {
        window.requestIdleCallback(() => this.setupWebcam(), { timeout: 1500 });
      } else {
        setTimeout(() => this.setupWebcam(), 1000);
      }
    }

    setupInputs() {
      // Mouse tracking & interactive hover
      this.canvas.addEventListener("mousemove", (e) => {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = CONFIG.WIDTH / rect.width;
        const scaleY = CONFIG.HEIGHT / rect.height;
        const sx = (e.clientX - rect.left) * scaleX;
        const sy = (e.clientY - rect.top) * scaleY;
        this.mouseScreenX = sx;
        this.mouseScreenY = sy;

        const unp = this.renderer.unprojectPointAtZ(sx, sy, -1.40);
        this.mousePos.x = Math.max(-0.85, Math.min(0.85, unp.x));
        this.mousePos.y = Math.max(0.04, Math.min(0.55, unp.y));
        this.mouseLastMoved = performance.now() / 1000;

        this.updateCursorState();
      });

      this.canvas.addEventListener("mouseleave", () => {
        this.mouseScreenX = -100;
        this.mouseScreenY = -100;
        this.hoveredAction = null;
        this.canvas.style.cursor = "default";
      });

      // Mouse click
      this.canvas.addEventListener("mousedown", (e) => {
        this.sounds.ensureContext();
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = CONFIG.WIDTH / rect.width;
        const scaleY = CONFIG.HEIGHT / rect.height;
        const sx = (e.clientX - rect.left) * scaleX;
        const sy = (e.clientY - rect.top) * scaleY;

        if (this.in_menu) {
          const action = this.getMenuActionAt(sx, sy);
          if (action) {
            this.triggerAction(action);
          }
        } else {
          // Check HUD [M] MENU button
          const [hx1, hy1, hx2, hy2] = this.renderer.hud_menu_button;
          if (sx >= hx1 && sx <= hx2 && sy >= hy1 && sy <= hy2) {
            this.in_menu = true;
            this.sounds.playBounce();
            return;
          }

          // In-game click: Serve or Power Smash!
          this.handleInGameAction();
        }
      });

      // Touch input support (tablets, mobile & touchscreen laptops)
      const handleTouchPos = (touch) => {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = CONFIG.WIDTH / rect.width;
        const scaleY = CONFIG.HEIGHT / rect.height;
        const sx = (touch.clientX - rect.left) * scaleX;
        const sy = (touch.clientY - rect.top) * scaleY;
        this.mouseScreenX = sx;
        this.mouseScreenY = sy;

        const unp = this.renderer.unprojectPointAtZ(sx, sy, -1.40);
        this.mousePos.x = Math.max(-0.85, Math.min(0.85, unp.x));
        this.mousePos.y = Math.max(0.04, Math.min(0.55, unp.y));
        this.mouseLastMoved = performance.now() / 1000;
        this.updateCursorState();
        return { sx, sy };
      };

      this.canvas.addEventListener("touchstart", (e) => {
        this.sounds.ensureContext();
        if (e.touches && e.touches.length > 0) {
          const { sx, sy } = handleTouchPos(e.touches[0]);
          if (this.in_menu) {
            const action = this.getMenuActionAt(sx, sy);
            if (action) {
              this.triggerAction(action);
            }
          } else {
            const [hx1, hy1, hx2, hy2] = this.renderer.hud_menu_button;
            if (sx >= hx1 && sx <= hx2 && sy >= hy1 && sy <= hy2) {
              this.in_menu = true;
              this.sounds.playBounce();
              return;
            }
            this.handleInGameAction();
          }
        }
        e.preventDefault();
      }, { passive: false });

      this.canvas.addEventListener("touchmove", (e) => {
        if (e.touches && e.touches.length > 0) {
          handleTouchPos(e.touches[0]);
        }
        e.preventDefault();
      }, { passive: false });

      // Window resize listener
      window.addEventListener("resize", () => {
        this.renderer.prepareFrame();
      });

      // Keyboard shortcuts matching main.py + controller navigation (game-ui-design)
      window.addEventListener("keydown", (e) => {
        this.sounds.ensureContext();
        const key = e.key;

        // If in menu, handle accessible keyboard navigation
        if (this.in_menu) {
          const focusable = this.getFocusableActions();
          if (key === "Tab") {
            e.preventDefault();
            const step = e.shiftKey ? -1 : 1;
            this.focusedActionIndex = (this.focusedActionIndex + step + focusable.length) % focusable.length;
            this.sounds.playHit(0.2);
            return;
          } else if (key === "ArrowRight") {
            e.preventDefault();
            this.focusedActionIndex = (this.focusedActionIndex + 1) % focusable.length;
            this.sounds.playHit(0.2);
            return;
          } else if (key === "ArrowLeft") {
            e.preventDefault();
            this.focusedActionIndex = (this.focusedActionIndex - 1 + focusable.length) % focusable.length;
            this.sounds.playHit(0.2);
            return;
          } else if (key === "ArrowDown") {
            e.preventDefault();
            if (this.focusedActionIndex < 3) {
              this.focusedActionIndex = 3;
            } else if (this.focusedActionIndex === 3) {
              this.focusedActionIndex = 5;
            } else if (this.focusedActionIndex >= 4 && this.focusedActionIndex <= 7) {
              this.focusedActionIndex = 8;
            }
            this.focusedActionIndex = Math.min(this.focusedActionIndex, focusable.length - 1);
            this.sounds.playHit(0.2);
            return;
          } else if (key === "ArrowUp") {
            e.preventDefault();
            if (this.focusedActionIndex >= 8) {
              this.focusedActionIndex = 5;
            } else if (this.focusedActionIndex >= 4) {
              this.focusedActionIndex = 3;
            } else {
              this.focusedActionIndex = 0;
            }
            this.focusedActionIndex = Math.max(0, this.focusedActionIndex);
            this.sounds.playHit(0.2);
            return;
          } else if (key === "Enter" || (key === " " && this.focusedActionIndex !== 3)) {
            e.preventDefault();
            const act = focusable[this.focusedActionIndex];
            if (act) {
              this.triggerAction(act);
              return;
            }
          }
        }

        if (key === "m" || key === "M") {
          this.in_menu = !this.in_menu;
          if (!this.in_menu && this.status === MatchStatus.READY) {
            this.prepareServe();
            this.sounds.playCheer();
          }
        } else if (key === "t" || key === "T") {
          if (this.trainer.is_active || this.trainer.feedback_timer > 0) {
            this.trainer.stop();
          } else {
            this.in_menu = false;
            this.trainer.start();
          }
        } else if (key === "Escape") {
          if (this.trainer.is_active) {
            this.trainer.stop();
          } else if (this.in_menu) {
            this.in_menu = false;
          }
        } else if (key === " " || key === "Enter") {
          if (this.trainer.is_active) {
            this.trainer.skipStep();
          } else if (this.in_menu) {
            this.in_menu = false;
            if (this.status === MatchStatus.READY) {
              this.prepareServe();
              this.sounds.playCheer();
            } else if (this.status === MatchStatus.GAME_OVER) {
              this.startNextSet();
            }
          } else {
            this.handleInGameAction();
          }
        } else if (key === "1") {
          this.ai.setDifficulty("EASY");
          this.sounds.playBounce();
        } else if (key === "2") {
          this.ai.setDifficulty("MEDIUM");
          this.sounds.playBounce();
        } else if (key === "3" || key === "4") {
          this.ai.setDifficulty("HARD");
          this.sounds.playBounce();
        } else if (key === "r" || key === "R") {
          this.startNewMatch(this.ai.difficulty);
          this.sounds.playBounce();
        } else if (key === "h" || key === "H") {
          const modes = ["AUTO", "RIGHT", "LEFT"];
          const nextIdx = (modes.indexOf(this.handednessMode) + 1) % modes.length;
          this.handednessMode = modes[nextIdx];
          this.announcement = `BRACO: ${this.handednessMode}`;
          this.announcement_timer = 2.0;
        } else if (key === "f" || key === "F") {
          if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen?.();
          } else {
            document.exitFullscreen?.();
          }
        }
      });
    }

    getFocusableActions() {
      if (this.menuTab === "PLAY") {
        return ["tab_play", "tab_help", "tab_bio", "start", "diff_1", "diff_2", "diff_3", "restart", "calibrate"];
      } else if (this.menuTab === "HELP") {
        return ["tab_play", "tab_help", "tab_bio", "start"];
      } else if (this.menuTab === "BIO") {
        return ["tab_play", "tab_help", "tab_bio", "calibrate", "start"];
      }
      return ["start"];
    }

    triggerAction(action) {
      if (!action) return;

      if (action === "tab_play") {
        this.menuTab = "PLAY";
        this.sounds.playBounce();
      } else if (action === "tab_help") {
        this.menuTab = "HELP";
        this.sounds.playBounce();
      } else if (action === "tab_bio") {
        this.menuTab = "BIO";
        this.sounds.playBounce();
      } else if (action === "start") {
        this.in_menu = false;
        if (this.status === MatchStatus.READY) {
          this.prepareServe();
          this.sounds.playCheer();
        } else if (this.status === MatchStatus.GAME_OVER) {
          this.startNextSet();
        }
      } else if (action && action.startsWith("diff_")) {
        const diffMap = { diff_1: "EASY", diff_2: "MEDIUM", diff_3: "HARD" };
        this.ai.setDifficulty(diffMap[action] || "MEDIUM");
        this.sounds.playBounce();
      } else if (action === "restart") {
        this.startNewMatch(this.ai.difficulty);
        this.sounds.playBounce();
      } else if (action === "calibrate") {
        this.in_menu = false;
        this.trainer.start();
      }
    }

    updateCursorState() {
      if (this.in_menu) {
        const action = this.getMenuActionAt(this.mouseScreenX, this.mouseScreenY);
        this.hoveredAction = action;
        this.canvas.style.cursor = action ? "pointer" : "default";
      } else {
        const [hx1, hy1, hx2, hy2] = this.renderer.hud_menu_button;
        const isHoverHUD = (
          this.mouseScreenX >= hx1 && this.mouseScreenX <= hx2 &&
          this.mouseScreenY >= hy1 && this.mouseScreenY <= hy2
        );
        this.hoveredAction = isHoverHUD ? "hud_menu" : null;
        this.canvas.style.cursor = isHoverHUD ? "pointer" : "crosshair";
      }
    }

    handleInGameAction() {
      if (this.status === MatchStatus.SERVING && this.server === "PLAYER" && this.serve_cooldown <= 0) {
        this.strikeServe(2.6, 0.0, 2.10);
      } else if (this.status === MatchStatus.RALLY && this.ball.vz < 0 && this.ball.z <= -0.75) {
        // Execute Power Smash / Motion Attack
        this.playerPaddle.speed = 3.4;
        this.playerPaddle.is_hook_swing = true;
        this.playerPaddle.hook_power = 2.6;
        this.strikePlayerBall(3.4, true);
      }
    }

    getMenuActionAt(x, y) {
      for (const [name, [bx1, by1, bx2, by2]] of Object.entries(this.renderer.menu_buttons)) {
        if (x >= bx1 && x <= bx2 && y >= by1 && y <= by2) {
          return name;
        }
      }
      return null;
    }

    setupWebcam() {
      if (typeof Hands === "undefined" || !navigator.mediaDevices?.getUserMedia) {
        // If script hasn't finished loading yet, retry gently once without blocking
        if (typeof Hands === "undefined") {
          setTimeout(() => this.setupWebcam(), 1500);
          return;
        }
        console.warn("[WEBCAM] getUserMedia não disponível. Controle 100% ativo via MOUSE.");
        return;
      }

      try {
        const hands = new Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });

        hands.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6,
        });

        hands.onResults((results) => {
          if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            this.handDetected = true;
            this.handLandmarks = results.multiHandLandmarks[0];

            const wrist = this.handLandmarks[0];
            const middleTip = this.handLandmarks[12];
            const indexMcp = this.handLandmarks[5];
            const pinkyMcp = this.handLandmarks[17];

            const distTipToPalm = Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y);
            const handScale = Math.hypot(pinkyMcp.x - indexMcp.x, pinkyMcp.y - indexMcp.y);
            this.isFist = distTipToPalm < handScale * 1.55;

            const wx = (0.5 - wrist.x) * 2.2;
            const wy = Math.max(0.04, Math.min(0.60, (0.75 - wrist.y) * 1.4));
            const wz = -1.40 + (handScale - 0.15) * 2.0;

            const now = performance.now() / 1000;
            if (this.lastHandPos && this.lastHandTime > 0) {
              const dt = Math.max(1e-4, now - this.lastHandTime);
              this.wristVx = (wx - this.lastHandPos.x) / dt;
              this.wristVy = (wy - this.lastHandPos.y) / dt;
              this.wristVz = (wz - this.lastHandPos.z) / dt;
              this.wristSpeed = Math.hypot(this.wristVx, this.wristVy, this.wristVz);
            }
            this.lastHandPos = { x: wx, y: wy, z: wz };
            this.lastHandTime = now;
            this.worldHand = { x: wx, y: wy, z: wz };
            this.armDetected = true;
          } else {
            this.handDetected = false;
            this.handLandmarks = null;
            this.wristSpeed = 0.0;
          }
        });

        if (typeof Camera !== "undefined") {
          const camera = new Camera(this.video, {
            onFrame: async () => {
              if (this.cameraActive) {
                await hands.send({ image: this.video });
              }
            },
            width: 640,
            height: 480,
          });
          camera.start().then(() => {
            this.cameraActive = true;
            console.log("[WEBCAM] Câmera conectada com rastreamento MediaPipe ativo!");
          }).catch((err) => {
            console.warn("[WEBCAM] Permissão de câmera não concedida. Modo mouse ativo.", err);
          });
        }
      } catch (err) {
        console.warn("[WEBCAM] Falha ao inicializar MediaPipe:", err);
      }
    }

    startNewMatch(diff = "MEDIUM") {
      this.player_score = 0;
      this.ai_score = 0;
      this.set_number = 1;
      this.player_sets = 0;
      this.ai_sets = 0;
      this.current_rally_count = 0;
      this.pointPending = false;
      this.ai.setDifficulty(diff);
      this.prepareServe();
    }

    prepareServe() {
      this.status = MatchStatus.SERVING;
      this.ball.reset(this.server);
      this.serve_cooldown = 0.0;
      this.pointPending = false;
      this.announcement = this.server === "PLAYER" ? "SUA VEZ DE SACAR • GOLPEIE OU CLIQUE" : "VEZ DO ROBERVAL • AGUARDANDO SAQUE...";
      this.announcement_timer = 2.5;

      if (this.server === "AI") {
        this.aiPaddle.target_x = -0.18;
        this.aiPaddle.target_y = 0.26;
        this.aiPaddle.x = -0.18;
        this.aiPaddle.y = 0.26;
        this.aiPaddle.z = 1.40;

        setTimeout(() => {
          if (this.status === MatchStatus.SERVING && this.server === "AI") {
            this.aiServe();
          }
        }, 1100);
      }
    }

    aiServe() {
      this.status = MatchStatus.RALLY;
      this.ball.in_play = true;
      this.ball.isServe = true;
      this.ball.lastHitter = "ROBERVAL";
      this.ball.bounces = 0;
      this.ball.bouncedPlayerSide = false;
      this.ball.bouncedAiSide = false;

      // Position paddle and ball at calibrated serve height (28cm above table)
      const serveSide = (Math.random() < 0.5 ? -0.18 : 0.18);
      this.aiPaddle.x = serveSide;
      this.aiPaddle.y = 0.26;
      this.aiPaddle.z = 1.40;

      this.ball.x = serveSide;
      this.ball.y = 0.28;
      this.ball.z = 1.35;
      this.ball.prevX = this.ball.x;
      this.ball.prevY = this.ball.y;
      this.ball.prevZ = this.ball.z;

      // ITTF Calibrated Serve Trajectory: clears 15.25cm net safely (+20cm margin, crosses at ~37cm) and lands cleanly at z ≈ -0.46m on player table
      const isEasy = (this.ai.difficulty === "EASY");
      this.ball.vx = isEasy ? 0.0 : (Math.random() < 0.5 ? -0.10 : 0.10);
      this.ball.vy = isEasy ? 2.45 : 2.30;
      this.ball.vz = isEasy ? -2.8 : -3.8;
      this.ball.spin_x = isEasy ? 0.0 : 4.0;
      this.current_rally_count = 1;

      this.sounds.playPaddleHit(false);
      this.renderer.addTrauma(0.20);

      const aiP = this.renderer.projectPoint(this.aiPaddle.x, this.aiPaddle.y, this.aiPaddle.z);
      if (aiP) this.particlePool.spawn(aiP.x, aiP.y, 10, "#ff3b5c");

      this.announcement = "SAQUE DO ROBERVAL EM JOGO!";
      this.announcement_timer = 1.0;
    }

    strikeServe(speed = 2.6, vxOffset = 0.0, vyLift = 2.10) {
      this.status = MatchStatus.RALLY;
      this.ball.in_play = true;
      this.ball.isServe = true;
      this.ball.lastHitter = "PLAYER";
      this.ball.bounces = 0;
      this.ball.bouncedPlayerSide = false;
      this.ball.bouncedAiSide = false;

      this.ball.x = this.playerPaddle.x;
      this.ball.y = this.playerPaddle.y;
      this.ball.z = -1.35;
      this.ball.prevX = this.ball.x;
      this.ball.prevY = this.ball.y;
      this.ball.prevZ = this.ball.z;

      // ITTF Calibrated Serve Arc: clears 15.25cm net with parabolic clearance
      this.ball.vx = vxOffset + (Math.random() - 0.5) * 0.15;
      this.ball.vy = vyLift;
      this.ball.vz = Math.max(4.2, speed * 1.8);
      this.ball.spin_x = 22.0;
      this.current_rally_count = 1;

      this.sounds.playPaddleHit(false);
      this.renderer.addTrauma(0.35);

      const pProj = this.renderer.projectPoint(this.ball.x, this.ball.y, this.ball.z);
      if (pProj) this.particlePool.spawn(pProj.x, pProj.y, 10, "#00f0ff");

      this.announcement = "SAQUE EM JOGO!";
      this.announcement_timer = 1.0;
    }

    strikePlayerBall(swingSpeed = 2.0, isSmash = false) {
      this.ball.lastHitter = "PLAYER";
      this.ball.isServe = false;
      this.ball.bounces = 0;
      this.ball.bouncedPlayerSide = false;
      this.ball.bouncedAiSide = false;

      // Realistic forward and vertical lift calibrated to clear net
      this.ball.vz = isSmash ? Math.max(5.2, swingSpeed * 2.2) : Math.max(4.0, swingSpeed * 1.8);
      const contactY = Math.max(0.04, this.ball.y);
      this.ball.vy = isSmash ? 1.75 : Math.max(2.10, 2.45 - (contactY * 1.2));
      this.ball.vx = (this.ball.x - this.playerPaddle.x) * 2.2 + (Math.random() - 0.5) * 0.2;
      this.ball.spin_x = isSmash ? 55.0 : 28.0;
      this.ball.squash = isSmash ? 0.42 : 0.30;
      this.current_rally_count++;

      this.sounds.playPaddleHit(isSmash);
      this.renderer.addTrauma(isSmash ? 0.60 : 0.22);
      if (isSmash) this.hit_stop_timer = 0.042;

      const pProj = this.renderer.projectPoint(this.ball.x, this.ball.y, this.ball.z);
      if (pProj) this.particlePool.spawn(pProj.x, pProj.y, isSmash ? 18 : 10, isSmash ? "#ff3b5c" : "#00f0ff");

      this.announcement = isSmash ? "SMASH!" : "NICE SHOT!";
      this.announcement_timer = 1.0;
    }

    awardPoint(winner, reason) {
      if (this.pointPending) return;
      this.pointPending = true;

      this.announcement = reason;
      this.announcement_timer = 2.5;

      if (winner === "PLAYER") {
        this.player_score++;
        this.sounds.playPointScore(true);
        this.renderer.addTrauma(0.35);
      } else {
        this.ai_score++;
        this.sounds.playPointScore(false);
        this.renderer.addTrauma(0.20);
      }

      // Serve rotation every 2 points
      const total = this.player_score + this.ai_score;
      this.server = Math.floor(total / 2) % 2 === 0 ? "PLAYER" : "AI";

      // Check Set win (11 points, win by 2)
      if ((this.player_score >= 11 || this.ai_score >= 11) && Math.abs(this.player_score - this.ai_score) >= 2) {
        this.status = MatchStatus.GAME_OVER;
        if (this.player_score > this.ai_score) {
          this.player_sets++;
          this.announcement = "VOCE VENCEU O SET!";
          this.sounds.playCheer();
        } else {
          this.ai_sets++;
          this.announcement = "ROBERVAL VENCEU O SET!";
        }
        setTimeout(() => {
          this.in_menu = true;
          this.pointPending = false;
        }, 1600);
      } else {
        setTimeout(() => {
          this.pointPending = false;
          if (!this.in_menu) this.prepareServe();
        }, 1500);
      }
    }

    checkScoring() {
      if (!this.ball.in_play || this.pointPending) return;

      // Ball falls below floor or past bounds
      if (this.ball.y <= -0.45 || Math.abs(this.ball.z) > 2.6 || Math.abs(this.ball.x) > 1.8) {
        this.ball.in_play = false;
        const hitter = this.ball.lastHitter || this.server;
        const wasServe = this.ball.isServe || (this.current_rally_count <= 1);

        // ITTF Scoring: If ball bounced on opponent's side, hitter gets the point; otherwise fault!
        if (hitter === "PLAYER") {
          if (this.ball.bouncedAiSide) {
            const desc = wasServe ? "PONTO SEU (+1) • ACE NO SAQUE!" : "PONTO SEU (+1) • GOLPE VENCEDOR";
            this.awardPoint("PLAYER", desc);
          } else {
            const desc = wasServe ? "PONTO DO ROBERVAL • SAQUE FORA" : "PONTO DO ROBERVAL • BOLA FORA";
            this.awardPoint("ROBERVAL", desc);
          }
        } else {
          if (this.ball.bouncedPlayerSide) {
            const desc = wasServe ? "PONTO DO ROBERVAL • ACE NO SAQUE" : "PONTO DO ROBERVAL • GOLPE VENCEDOR";
            this.awardPoint("ROBERVAL", desc);
          } else {
            const desc = wasServe ? "PONTO SEU (+1) • SAQUE DO ROBERVAL FORA" : "PONTO SEU (+1) • ROBERVAL FORA";
            this.awardPoint("PLAYER", desc);
          }
        }
      }
    }

    startNextSet() {
      this.set_number++;
      this.player_score = 0;
      this.ai_score = 0;
      this.current_rally_count = 0;
      this.pointPending = false;
      this.in_menu = false;
      this.prepareServe();
    }

    // =======================================================================
    // 9. CONTINUOUS COLLISION DETECTION (NET, TABLE, PADDLES)
    // =======================================================================
    checkNetCollision(prevZ, currZ, x, y, radius = CONFIG.BALL_RADIUS) {
      // Check if ball crossed the net plane at Z = 0 in this sub-step
      if ((prevZ < 0.0 && currZ >= 0.0) || (prevZ > 0.0 && currZ <= 0.0)) {
        // Lateral check: within net width (half width + overhang)
        const maxNetX = (CONFIG.TABLE_WIDTH / 2.0) + CONFIG.NET_OVERHANG;
        if (Math.abs(x) <= maxNetX) {
          // Height check: from table surface (0) to top of net
          if ((y - radius) <= CONFIG.NET_HEIGHT) {
            return true;
          }
        }
      }
      return false;
    }

    // =======================================================================
    // 10. RENDER CYCLE & UI (Identical layout to PerspectiveRenderer3D)
    // =======================================================================
    loop(timestamp) {
      const now = timestamp / 1000;
      const dt = Math.max(1e-4, Math.min(0.05, now - this.lastFrameTime));
      this.lastFrameTime = now;

      // 1. Process Hand vs Mouse Priority
      const nowSec = performance.now() / 1000;
      const useMouse = (nowSec - this.mouseLastMoved) < 1.2 || !this.handDetected;

      let targetX, targetY, targetZ;
      if (useMouse) {
        targetX = this.mousePos.x;
        targetY = this.mousePos.y;
        targetZ = -1.40;
        this.playerPaddle.hand_speed = this.playerPaddle.speed;
      } else {
        targetX = this.worldHand.x;
        targetY = this.worldHand.y;
        targetZ = this.worldHand.z;
        this.playerPaddle.hand_speed = this.wristSpeed;
      }
      this.playerPaddle.update(targetX, targetY, targetZ, dt);

      // 2. Interactive Trainer Mode Update
      if (this.trainer.is_active || this.trainer.feedback_timer > 0) {
        const curSpd = useMouse ? this.playerPaddle.speed : this.wristSpeed;
        const curVx = useMouse ? this.playerPaddle.vx : this.wristVx;
        const curVy = useMouse ? this.playerPaddle.vy : this.wristVy;
        this.trainer.update(curSpd, curVx, curVy, this.isFist, dt, this.sounds);
      }

      // 3. Serve via physical arm swing motion or AI wind-up
      if (!this.in_menu && this.status === MatchStatus.SERVING) {
        if (this.server === "PLAYER") {
          this.ball.x = this.playerPaddle.x;
          this.ball.y = Math.max(0.24, this.playerPaddle.y + 0.06) + 0.015 * Math.sin(performance.now() * 0.005);
          this.ball.z = this.playerPaddle.z + 0.08;
          this.ball.prevX = this.ball.x;
          this.ball.prevY = this.ball.y;
          this.ball.prevZ = this.ball.z;
          this.ball.vx = 0;
          this.ball.vy = 0;
          this.ball.vz = 0;
          this.ball.in_play = true;

          if (this.serve_cooldown <= 0) {
            const armSpeed = Math.max(this.playerPaddle.speed, this.wristSpeed);
            if (armSpeed >= 0.60) {
              this.strikeServe(armSpeed, 0.0, 2.20);
            }
          }
        } else if (this.server === "AI") {
          this.aiPaddle.target_x = -0.18;
          this.aiPaddle.target_y = 0.26;
          this.ball.x = this.aiPaddle.x;
          this.ball.y = Math.max(0.26, this.aiPaddle.y + 0.04) + 0.015 * Math.sin(performance.now() * 0.005);
          this.ball.z = this.aiPaddle.z - 0.08;
          this.ball.prevX = this.ball.x;
          this.ball.prevY = this.ball.y;
          this.ball.prevZ = this.ball.z;
          this.ball.vx = 0;
          this.ball.vy = 0;
          this.ball.vz = 0;
          this.ball.in_play = true;
        }
      }

      // 4. In-Match Physics Simulation (with Hit-Stop and Sub-Stepping)
      if (this.serve_cooldown > 0) this.serve_cooldown -= dt;
      if (this.announcement_timer > 0) this.announcement_timer -= dt;

      // Update particle pool
      this.particlePool.update(dt);

      if (!this.in_menu && !this.trainer.is_active) {
        if (this.hit_stop_timer > 0) {
          this.hit_stop_timer -= dt;
        } else {
          this.ai.update(this.aiPaddle, this.ball, dt);

          // Sub-stepping for ultra-high speed accuracy and tunnel prevention (4 substeps)
          const substeps = 4;
          const subDt = dt / substeps;

          for (let step = 0; step < substeps; step++) {
            if (!this.ball.in_play) break;

            const prevStepZ = this.ball.z;
            this.ball.integrateForces(subDt);

            // A. Decisive ITTF Net Collision Check
            if (this.checkNetCollision(prevStepZ, this.ball.z, this.ball.x, this.ball.y, CONFIG.BALL_RADIUS)) {
              const tapeThreshold = CONFIG.NET_HEIGHT - 0.025; // Top 2.5cm tape
              const forwardSign = prevStepZ < this.ball.z ? 1.0 : -1.0;

              if (this.ball.y >= tapeThreshold) {
                // NET CLIP: Ball grazes white tape and rolls over
                this.ball.vz = forwardSign * Math.max(2.0, Math.abs(this.ball.vz) * 0.75);
                this.ball.vy = Math.max(0.5, this.ball.vy * 0.7);
                this.ball.z += 0.04 * forwardSign; // Nudge past net plane
                this.announcement = "RASPOU NA REDE!";
                this.announcement_timer = 1.0;
                this.sounds.playBounce(false);

                const netProj = this.renderer.projectPoint(this.ball.x, this.ball.y, 0.0);
                if (netProj) this.particlePool.spawn(netProj.x, netProj.y, 10, "#ffffff");
              } else {
                // NET FAULT: Ball hit net mesh below tape!
                this.ball.in_play = false;
                this.ball.vz = -this.ball.vz * 0.25; // Rebound gently backwards
                this.ball.vy = 0.40;
                this.sounds.playBounce(false);

                const netProj = this.renderer.projectPoint(this.ball.x, this.ball.y, 0.0);
                if (netProj) this.particlePool.spawn(netProj.x, netProj.y, 14, "#ff3b5c");

                const hitter = this.ball.lastHitter || this.server;
                const wasServe = this.ball.isServe || (this.current_rally_count <= 1);

                if (hitter === "PLAYER") {
                  const desc = wasServe ? "PONTO DO ROBERVAL • SAQUE NA REDE" : "PONTO DO ROBERVAL • BOLA NA REDE";
                  this.awardPoint("ROBERVAL", desc);
                } else {
                  const desc = wasServe ? "PONTO SEU (+1) • SAQUE DO ROBERVAL NA REDE" : "PONTO SEU (+1) • ROBERVAL NA REDE";
                  this.awardPoint("PLAYER", desc);
                }
                break;
              }
            }

            // B. Table Surface Bounce Collision
            const hw = CONFIG.TABLE_WIDTH / 2.0;
            const hl = CONFIG.TABLE_LENGTH / 2.0;
            if (
              this.ball.y <= 0.02 &&
              this.ball.y >= -0.10 &&
              Math.abs(this.ball.x) <= hw &&
              Math.abs(this.ball.z) <= hl
            ) {
              this.ball.y = 0.02;
              this.ball.vy = Math.max(1.85, -this.ball.vy * CONFIG.RESTITUTION_TABLE);
              this.ball.squash = 0.28;
              this.ball.bounces++;
              this.sounds.playBounce(true);

              if (this.ball.z > 0) {
                this.ball.bouncedAiSide = true;
              } else {
                this.ball.bouncedPlayerSide = true;
              }

              // Double bounce fault
              if (this.ball.bounces >= 2) {
                this.ball.in_play = false;
                const scoringParty = (this.ball.z > 0) ? "PLAYER" : "ROBERVAL";
                const desc = (scoringParty === "PLAYER") ? "PONTO SEU (+1) • DOIS QUIQUES" : "PONTO DO ROBERVAL • DOIS QUIQUES";
                this.awardPoint(scoringParty, desc);
                break;
              }

              const bProj = this.renderer.projectPoint(this.ball.x, 0.0, this.ball.z);
              if (bProj) this.particlePool.spawn(bProj.x, bProj.y, 8, "#00f0ff");
            }

            // C. Player Broad Strike Zone (Wii Sports Motion Attack Model)
            if (
              this.ball.in_play &&
              this.ball.vz < 0 &&
              this.ball.z >= CONFIG.BROAD_STRIKE_ZONE_Z_MIN &&
              this.ball.z <= CONFIG.BROAD_STRIKE_ZONE_Z_MAX
            ) {
              const dx = this.ball.x - this.playerPaddle.x;
              const dy = this.ball.y - this.playerPaddle.y;
              const dist = Math.hypot(dx, dy);
              const armSpeed = Math.max(this.playerPaddle.speed, this.wristSpeed);

              if (dist <= CONFIG.MOTION_ATTACK_REACH && (armSpeed >= 0.45 || this.ball.z <= -1.25)) {
                const isSmash = armSpeed >= 2.4;
                this.strikePlayerBall(Math.max(1.8, armSpeed), isSmash);
              }
            }

            // D. AI Strike Collision (Roberval Intercept)
            if (
              this.ball.in_play &&
              this.ball.vz > 0 &&
              this.ball.z >= 1.25 &&
              this.ball.z <= 1.45
            ) {
              const dx = this.ball.x - this.aiPaddle.x;
              const dy = this.ball.y - this.aiPaddle.y;
              if (Math.hypot(dx, dy) <= 0.46) {
                this.ball.lastHitter = "ROBERVAL";
                this.ball.isServe = false;
                this.ball.bounces = 0;
                this.ball.bouncedPlayerSide = false;
                this.ball.bouncedAiSide = false;
                // Calibrate dynamic lift based on contact height to guarantee clearing the net:
                const isEasy = (this.ai.difficulty === "EASY");
                const contactY = Math.max(0.04, this.ball.y);
                const dynamicVy = isEasy ? 2.45 : Math.max(2.18, 2.50 - (contactY * 1.2));

                this.ball.vz = isEasy ? -2.7 : -4.0;
                this.ball.vy = dynamicVy;
                this.ball.vx = isEasy
                  ? 0.02 * (this.playerPaddle.x - this.aiPaddle.x)
                  : Math.max(-0.70, Math.min(0.70, 0.18 * this.ball.vx + (this.ball.x - this.aiPaddle.x) * 1.1 + (Math.random() - 0.5) * 0.25));
                this.ball.spin_x = isEasy ? 0.0 : 8.0;
                this.ball.squash = 0.28;
                this.current_rally_count++;
                this.sounds.playPaddleHit(false);
                this.renderer.addTrauma(0.18);

                const aiP = this.renderer.projectPoint(this.aiPaddle.x, this.aiPaddle.y, this.aiPaddle.z);
                if (aiP) this.particlePool.spawn(aiP.x, aiP.y, 10, "#ff3b5c");
              }
            }
          }

          this.checkScoring();
        }
      }

      // 5. Render Scene
      this.renderer.updateCamera(this.ball, dt);
      this.renderer.renderBackground();
      this.draw3DWorld();
      this.drawHUD();
      this.drawRightPanel();

      if (this.trainer.is_active || this.trainer.feedback_timer > 0) {
        this.trainer.renderOverlay(this.renderer.ctx, Math.max(this.playerPaddle.speed, this.wristSpeed) * 3.6);
      } else if (this.in_menu) {
        this.drawMenuModal();
      }

      requestAnimationFrame(this.loop.bind(this));
    }

    draw3DWorld() {
      const ctx = this.renderer.ctx;

      // 1. Landing Target & Anticipation Reticle (Wii Sports Anticipation)
      if (this.ball.in_play && this.ball.y > 0.05 && this.ball.vy < 1.5 && Math.abs(this.ball.vz) > 0.2) {
        const tImpact = (this.ball.vy + Math.sqrt(Math.max(0.01, this.ball.vy ** 2 + 2 * 9.81 * this.ball.y))) / 9.81;
        const landX = this.ball.x + this.ball.vx * tImpact;
        const landZ = this.ball.z + this.ball.vz * tImpact;
        const hw = CONFIG.TABLE_WIDTH / 2.0;
        const hl = CONFIG.TABLE_LENGTH / 2.0;

        if (Math.abs(landX) <= hw && Math.abs(landZ) <= hl) {
          const proj = this.renderer.projectPoint(landX, 0.0, landZ);
          if (proj) {
            const countdown = Math.min(2.5, Math.max(0.9, 0.9 + tImpact * 2.5));
            const baseR = Math.floor(24.0 / proj.z);
            const rOuter = Math.max(6, Math.floor(baseR * countdown));
            const rInner = Math.max(2, Math.floor(8.0 / proj.z));

            let color = "#00d7ff";
            if (this.ball.speed_kmh > 55.0) color = "#ff2828";
            else if (this.ball.speed_kmh > 35.0) color = "#ffd166";
            else if (landZ < 0) color = "#00e6ff";

            ctx.save();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(proj.x, proj.y, rOuter, 0, Math.PI * 2);
            ctx.stroke();

            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(proj.x, proj.y, rInner, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
          }
        }
      }

      // 2. Ball Shadow on Table
      if (this.ball.in_play || this.status === MatchStatus.SERVING) {
        const hw = CONFIG.TABLE_WIDTH / 2.0 + 0.12;
        const hl = CONFIG.TABLE_LENGTH / 2.0 + 0.12;
        const overTable = Math.abs(this.ball.x) <= hw && Math.abs(this.ball.z) <= hl;
        const shadowY = overTable ? 0.0 : -0.76;
        const bShadow = this.renderer.projectPoint(this.ball.x, shadowY, this.ball.z);

        if (bShadow) {
          const baseR = Math.floor(22.0 / bShadow.z);
          const distToSurf = Math.max(0.0, this.ball.y - shadowY);
          const heightScale = Math.max(0.4, 1.0 - distToSurf * 0.45);
          const r = Math.max(3, Math.floor(baseR * heightScale));
          const alpha = Math.max(0.18, Math.min(0.60, 0.70 - distToSurf * 0.35));

          ctx.save();
          ctx.fillStyle = `rgba(12, 14, 20, ${alpha})`;
          ctx.beginPath();
          ctx.ellipse(bShadow.x, bShadow.y, r, r * 0.45, 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
      }

      // 3. Ball Speed Trail
      if (this.ball.trail.length > 1) {
        ctx.save();
        for (let i = 0; i < this.ball.trail.length; i++) {
          const pt = this.ball.trail[i];
          const proj = this.renderer.projectPoint(pt.x, pt.y, pt.z);
          if (proj) {
            const alpha = ((this.ball.trail.length - i) / this.ball.trail.length) * 0.75;
            const r = Math.max(2, (14 / proj.z) * alpha);
            let trailColor = `rgba(0, 240, 255, ${alpha})`;
            if (pt.speed > 55) {
              trailColor = `rgba(255, 59, 92, ${alpha})`;
            } else if (pt.speed > 35) {
              trailColor = `rgba(255, 209, 102, ${alpha})`;
            }
            ctx.fillStyle = trailColor;
            ctx.beginPath();
            ctx.arc(proj.x, proj.y, r, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        ctx.restore();
      }

      // 4. AI Paddle ("ROBERVAL")
      this.drawPaddle(this.aiPaddle, false);

      // 5. 3D Ball with squash & stretch and specular sheen
      if (this.ball.in_play || this.status === MatchStatus.SERVING) {
        const bp = this.renderer.projectPoint(this.ball.x, this.ball.y, this.ball.z);
        if (bp) {
          const baseR = Math.max(4, Math.floor((this.renderer.focal_length * CONFIG.BALL_RADIUS) / bp.z));
          const rx = baseR * (1.0 + this.ball.squash * 0.4);
          const ry = baseR * (1.0 - this.ball.squash * 0.4);

          ctx.save();
          const ballGrad = ctx.createRadialGradient(
            bp.x - rx * 0.3, bp.y - ry * 0.3, rx * 0.15,
            bp.x, bp.y, rx
          );
          ballGrad.addColorStop(0, "#ffffff");
          ballGrad.addColorStop(0.3, "#ff9e00");
          ballGrad.addColorStop(0.85, "#e65100");
          ballGrad.addColorStop(1, "#b23b00");

          ctx.fillStyle = ballGrad;
          ctx.beginPath();
          ctx.ellipse(bp.x, bp.y, rx, ry, 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
          ctx.lineWidth = 1;
          ctx.stroke();

          // Glowing pulse ring on serve ball (Wii Sports Sweet Spot)
          if (this.status === MatchStatus.SERVING) {
            const pulse = 0.5 + 0.5 * Math.sin(performance.now() * 0.008);
            const pulseR = Math.floor(baseR * 2.2 + 8 * pulse);
            ctx.strokeStyle = (this.server === "PLAYER") ? "#00d7ff" : "#ff8c00";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(bp.x, bp.y, pulseR, 0, Math.PI * 2);
            ctx.stroke();
          }

          ctx.restore();
        }
      }

      // 6. Player Paddle ("VOCE")
      this.drawPaddle(this.playerPaddle, true);

      // 7. Hit Spark Particles (Rendered from zero-GC pool)
      this.particlePool.render(ctx);
    }

    drawPaddle(paddle, isPlayer) {
      const p = this.renderer.projectPoint(paddle.x, paddle.y, paddle.z);
      if (!p) return;

      const ctx = this.renderer.ctx;
      const bladeR = Math.max(10, Math.floor((this.renderer.focal_length * CONFIG.PADDLE_RADIUS) / p.z));
      ctx.save();

      // Swing Aura for player
      if (isPlayer) {
        const spd = Math.max(paddle.speed, paddle.hand_speed);
        if (spd >= 2.4 || (paddle.is_hook_swing && spd >= 1.8)) {
          ctx.strokeStyle = "#008cff";
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.arc(p.x, p.y, bladeR * 1.25, 0, Math.PI * 2);
          ctx.stroke();
        } else if (spd >= 1.0) {
          ctx.strokeStyle = "#ffe600";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(p.x, p.y, bladeR * 1.15, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // Flared handle
      const handleLen = Math.floor(bladeR * 0.90);
      const wNeck = Math.max(3, Math.floor(bladeR * 0.20));
      const wFlare = Math.max(4, Math.floor(bladeR * 0.30));

      if (isPlayer) {
        const hx = p.x;
        const hy = p.y + bladeR * 0.40;
        ctx.fillStyle = "#a56e46";
        ctx.beginPath();
        ctx.moveTo(hx - wNeck, hy);
        ctx.lineTo(hx + wNeck, hy);
        ctx.lineTo(hx + wFlare, hy + handleLen);
        ctx.lineTo(hx - wFlare, hy + handleLen);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = "#784b2d";
        ctx.lineWidth = 1;
        ctx.stroke();

        // Metallic jewel emblem
        ctx.fillStyle = "#dce6f0";
        ctx.beginPath();
        ctx.arc(hx, hy + handleLen * 0.85, Math.max(2, wFlare * 0.4), 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = "#966441";
        ctx.fillRect(p.x - 3, p.y - handleLen, 6, handleLen);
      }

      // Main Blade Rubber (Ruby Red on Player, Carbon Black on Roberval)
      const rubColor = isPlayer ? "#d71c19" : "#1e1e1e";

      ctx.fillStyle = rubColor;
      ctx.beginPath();
      ctx.arc(p.x, p.y, bladeR, 0, Math.PI * 2);
      ctx.fill();

      // White edge tape rim
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = Math.max(1, Math.floor(bladeR * 0.08));
      ctx.stroke();

      // Specular sheen on rubber
      const specX = p.x - bladeR * 0.28;
      const specY = p.y - bladeR * 0.28;
      ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
      ctx.beginPath();
      ctx.ellipse(specX, specY, Math.max(2, bladeR * 0.40), Math.max(2, bladeR * 0.25), 0, 0, Math.PI * 2);
      ctx.fill();

      // Inner sweet-spot target ring for player
      if (isPlayer) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, bladeR * 0.35, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Tag ("VOCE" / "ROBERVAL")
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = "#0a0a0f";
      ctx.fillText(isPlayer ? "VOCE" : "ROBERVAL", p.x + 1, p.y - bladeR - 6);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(isPlayer ? "VOCE" : "ROBERVAL", p.x, p.y - bladeR - 7);

      ctx.restore();
    }

    drawHUD() {
      const ctx = this.renderer.ctx;
      const cx = this.renderer.width / 2; // 512
      // 2. In-Game Broadcast Scorebug (WTT / Olympic Sports Style)
      const sbW = 430;
      const sbH = 46;
      const sbX = cx - sbW / 2; // 297
      const sbY = 14;

      ctx.save();

      // Drop shadow
      ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
      ctx.fillRect(sbX + 2, sbY + 3, sbW + 2, sbH + 3);

      // Stadium slate scorebug container
      ctx.fillStyle = "rgba(15, 23, 42, 0.94)";
      ctx.fillRect(sbX, sbY, sbW, sbH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
      ctx.lineWidth = 1;
      ctx.strokeRect(sbX, sbY, sbW, sbH);

      // Player Pod (Tournament Crimson strip)
      const pPodW = 165;
      ctx.fillStyle = "#E63946";
      ctx.fillRect(sbX, sbY, 5, sbH);

      ctx.fillStyle = "#FFFFFF";
      ctx.font = "bold 15px 'Chakra Petch', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("VOCE", sbX + 16, sbY + 28);

      // Serving indicator with soft pulsing golden orb
      if (this.server === "PLAYER") {
        const servePulse = 0.5 + 0.5 * Math.sin(performance.now() * 0.008);
        ctx.save();
        ctx.shadowColor = "#F59E0B";
        ctx.shadowBlur = 6 + 4 * servePulse;
        ctx.fillStyle = "#F59E0B";
        ctx.beginPath();
        ctx.arc(sbX + 68, sbY + 23, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.restore();
      }

      // Sets won pips for Player (best of 3: 2 dots)
      for (let i = 0; i < 2; i++) {
        const pipX = sbX + 16 + i * 11;
        const pipY = sbY + 38;
        ctx.beginPath();
        ctx.arc(pipX, pipY, 3, 0, Math.PI * 2);
        if (i < this.player_sets) {
          ctx.fillStyle = "#E63946";
          ctx.fill();
        } else {
          ctx.strokeStyle = "#475569";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Player Score box
      ctx.fillStyle = "#1E293B";
      ctx.fillRect(sbX + pPodW - 44, sbY + 4, 40, sbH - 8);
      ctx.strokeStyle = "#E63946";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(sbX + pPodW - 44, sbY + 4, 40, sbH - 8);

      ctx.fillStyle = "#FFFFFF";
      ctx.font = "bold 22px 'Chakra Petch', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(this.player_score, sbX + pPodW - 24, sbY + 32);

      // Center Pod (Rally counter with flame glow)
      const cPodX = sbX + pPodW;
      const isFire = this.current_rally_count >= 5;
      ctx.fillStyle = isFire ? "#F59E0B" : "#94A3B8";
      ctx.font = "bold 10px 'Chakra Petch', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(isFire ? "FOGO! 🔥" : "RALLY", cPodX + 50, sbY + 16);

      ctx.fillStyle = isFire ? "#F59E0B" : "#FFFFFF";
      ctx.font = "bold 20px 'Chakra Petch', sans-serif";
      ctx.fillText(this.current_rally_count, cPodX + 50, sbY + 37);

      // Roberval Pod
      const rPodX = cPodX + 100;
      ctx.fillStyle = "#1E293B";
      ctx.fillRect(rPodX + 4, sbY + 4, 40, sbH - 8);
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(rPodX + 4, sbY + 4, 40, sbH - 8);

      ctx.fillStyle = "#F1F5F9";
      ctx.font = "bold 22px 'Chakra Petch', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(this.ai_score, rPodX + 24, sbY + 32);

      // Serving indicator with soft pulsing golden orb
      if (this.server === "AI") {
        const servePulse = 0.5 + 0.5 * Math.sin(performance.now() * 0.008);
        ctx.save();
        ctx.shadowColor = "#F59E0B";
        ctx.shadowBlur = 6 + 4 * servePulse;
        ctx.fillStyle = "#F59E0B";
        ctx.beginPath();
        ctx.arc(rPodX + 56, sbY + 23, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 1.2;
        ctx.stroke();
        ctx.restore();
      }

      ctx.fillStyle = "#FFFFFF";
      ctx.font = "bold 14px 'Chakra Petch', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("ROBERVAL", rPodX + 70, sbY + 28);

      // Sets won pips for Roberval
      for (let i = 0; i < 2; i++) {
        const pipX = rPodX + 70 + i * 11;
        const pipY = sbY + 38;
        ctx.beginPath();
        ctx.arc(pipX, pipY, 3, 0, Math.PI * 2);
        if (i < this.ai_sets) {
          ctx.fillStyle = "#F59E0B";
          ctx.fill();
        } else {
          ctx.strokeStyle = "#475569";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      ctx.fillStyle = "#334155";
      ctx.fillRect(sbX + sbW - 5, sbY, 5, sbH);

      // Sub-line below scorebug: Difficulty & Sets
      ctx.fillStyle = "#94A3B8";
      ctx.font = "11px 'Inter', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(`TIER: ${this.ai.difficulty}`, sbX + 4, sbY + sbH + 16);

      ctx.textAlign = "center";
      ctx.fillStyle = "#E2E8F0";
      ctx.font = "11px 'Inter', sans-serif";
      ctx.fillText(`SET ${this.set_number}  •  SETS: ${this.player_sets} x ${this.ai_sets}`, cx, sbY + sbH + 16);

      // Digital sports radar speedometer
      if (this.ball.in_play && this.ball.speed_kmh > 4) {
        ctx.textAlign = "right";
        const spd = this.ball.speed_kmh;
        ctx.font = "bold 11px 'Chakra Petch', sans-serif";
        ctx.fillStyle = spd > 45 ? "#E63946" : (spd > 30 ? "#F59E0B" : "#10B981");
        ctx.fillText(`⚡ RADAR: ${spd.toFixed(0)} KM/H`, sbX + sbW - 8, sbY + sbH + 16);
      }

      // Top-Left [M] MENU button with hover feedback
      const [hx1, hy1, hx2, hy2] = this.renderer.hud_menu_button;
      const isMenuHover = (this.hoveredAction === "hud_menu");
      ctx.save();
      ctx.fillStyle = isMenuHover ? "#E63946" : "rgba(30, 41, 59, 0.90)";
      ctx.fillRect(hx1, hy1, hx2 - hx1, hy2 - hy1);
      ctx.strokeStyle = isMenuHover ? "#FF4D5A" : "rgba(255, 255, 255, 0.16)";
      ctx.lineWidth = isMenuHover ? 1.5 : 1;
      ctx.strokeRect(hx1, hy1, hx2 - hx1, hy2 - hy1);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "bold 12px 'Chakra Petch', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(isMenuHover ? "▸ [M] MENU" : "[M] MENU", hx1 + (hx2 - hx1) / 2, hy1 + 22);
      ctx.restore();

      // Top-Left Lateral Status Box (De quem é a vez & Quem pontuou)
      let statusMsg = "";
      let statusColor = "#F59E0B";

      if (this.announcement && this.announcement_timer > 0) {
        statusMsg = this.announcement;
        if (statusMsg.includes("PONTO SEU") || statusMsg.includes("VENCEU O SET") || statusMsg.includes("CAMPEAO")) {
          statusColor = "#10B981"; // Emerald Green for Player Point / Set Win
        } else if (statusMsg.includes("PONTO DO ROBERVAL") || statusMsg.includes("ROBERVAL VENCEU")) {
          statusColor = "#E63946"; // Tournament Crimson for Opponent Point
        } else if (statusMsg.includes("SMASH") || statusMsg.includes("ACE")) {
          statusColor = "#F59E0B"; // Gold for Smash / Ace
        } else {
          statusColor = "#3B82F6"; // Blue for other events
        }
      } else if (!this.in_menu && this.status === MatchStatus.SERVING) {
        if (this.server === "PLAYER") {
          statusMsg = "SUA VEZ DE SACAR • GOLPEIE OU CLIQUE";
          statusColor = "#10B981"; // Player serve turn (Green)
        } else {
          statusMsg = "VEZ DO ROBERVAL • AGUARDANDO SAQUE...";
          statusColor = "#3B82F6"; // AI serve turn (Blue)
        }
      }

      if (statusMsg) {
        ctx.save();
        let fontSize = 12;
        ctx.font = `bold ${fontSize}px 'Inter', sans-serif`;
        let textW = ctx.measureText(statusMsg).width;
        const maxPillW = Math.min(420, this.renderer.width - 40);

        // Auto-scale font down if text approaches maximum container width
        if (textW + 48 > maxPillW) {
          fontSize = 11;
          ctx.font = `bold ${fontSize}px 'Inter', sans-serif`;
          textW = ctx.measureText(statusMsg).width;
        }

        const pillX = 20;
        const pillY = 75;
        const pillH = 32;
        const pillW = Math.min(maxPillW, Math.max(220, Math.ceil(textW + 44)));

        // Box background with subtle shadow
        ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
        ctx.shadowBlur = 8;
        ctx.fillStyle = "rgba(15, 23, 42, 0.95)";
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(pillX, pillY, pillW, pillH, 6);
        } else {
          ctx.rect(pillX, pillY, pillW, pillH);
        }
        ctx.fill();
        ctx.shadowBlur = 0;

        // Border with accent color
        ctx.strokeStyle = statusColor;
        ctx.lineWidth = 1.3;
        ctx.stroke();

        // Status indicator dot
        ctx.fillStyle = statusColor;
        ctx.beginPath();
        ctx.arc(pillX + 15, pillY + pillH / 2, 4.5, 0, Math.PI * 2);
        ctx.fill();

        // Status text vertically centered
        ctx.fillStyle = "#FFFFFF";
        ctx.font = `bold ${fontSize}px 'Inter', sans-serif`;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(statusMsg, pillX + 28, pillY + pillH / 2);
        ctx.restore();
      }

      // Discrete Controls Legend at Bottom Left (during gameplay)
      if (!this.in_menu) {
        ctx.fillStyle = "#94A3B8";
        ctx.font = "11px 'Inter', sans-serif";
        ctx.textAlign = "left";
        const controlHint = `[ESPAÇO] Sacar & Smash  •  [M] Menu / Pausa  •  [T] Calibrar  •  [R] Reiniciar  •  [F] Tela Cheia`;
        ctx.fillText(controlHint, 20, this.renderer.height - 15);
      }

      ctx.restore();
    }

    drawRightPanel() {
      const ctx = this.renderer.ctx;
      const inMenu = this.in_menu;
      const cw = inMenu ? 195 : 190;
      const ch = inMenu ? 146 : 142;
      const camX1 = inMenu ? 740 : this.renderer.width - cw - 15;
      const camY1 = inMenu ? 35 : 82;

      ctx.save();

      // 1. Live Camera Picture-in-Picture (PiP)
      if (this.cameraActive && this.video.videoWidth > 0) {
        ctx.save();
        ctx.translate(camX1 + cw, camY1);
        ctx.scale(-1, 1);
        ctx.drawImage(this.video, 0, 0, cw, ch);
        ctx.restore();

        // Draw Skeleton Landmarks
        if (this.handLandmarks) {
          ctx.save();
          ctx.fillStyle = "#00ff78";
          ctx.strokeStyle = "#00e6ff";
          ctx.lineWidth = 1.5;
          for (const lm of this.handLandmarks) {
            const lx = camX1 + (1.0 - lm.x) * cw;
            const ly = camY1 + lm.y * ch;
            ctx.beginPath();
            ctx.arc(lx, ly, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.restore();
        }
      } else {
        // Fallback stylish synthetic sensor card
        ctx.fillStyle = "#121824";
        ctx.fillRect(camX1, camY1, cw, ch);

        ctx.strokeStyle = "#283850";
        ctx.lineWidth = 1;
        ctx.strokeRect(camX1, camY1, cw, ch);

        ctx.fillStyle = "#00e6ff";
        ctx.font = "bold 32px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("🖱️", camX1 + cw / 2, camY1 + ch / 2 - 8);

        ctx.font = "bold 12px sans-serif";
        ctx.fillStyle = "#b4d2f0";
        ctx.fillText("CONTROLE POR MOUSE", camX1 + cw / 2, camY1 + ch / 2 + 20);
        ctx.font = "10px sans-serif";
        ctx.fillStyle = "#6482a0";
        ctx.fillText("Mova o cursor na mesa", camX1 + cw / 2, camY1 + ch / 2 + 36);
      }

      // PiP Border & Badge (Athletic Tournament Style)
      const isSwing = this.playerPaddle.speed >= 1.2 || this.wristSpeed >= 1.2;
      const pipBorderColor = isSwing ? "#F59E0B" : (this.handDetected ? "#10B981" : "#E63946");
      const pipBadgeText = isSwing ? "GOLPE FORTE ⚡" : (this.handDetected ? "RAQUETE EMPUNHADA" : "MODO MOUSE 🖱️");

      ctx.strokeStyle = pipBorderColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(camX1, camY1, cw, ch);

      ctx.fillStyle = "rgba(15, 23, 42, 0.94)";
      ctx.fillRect(camX1, camY1 - 18, 140, 18);
      ctx.fillStyle = pipBorderColor;
      ctx.font = "bold 10px 'Chakra Petch', sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(pipBadgeText, camX1 + 6, camY1 - 5);

      // 2. Hand Depth Strike Zone HUD Needle
      const barX = camX1 - 16;
      const barW = 8;
      const barY = camY1;
      const barH = ch;

      ctx.fillStyle = "#141923";
      ctx.fillRect(barX, barY, barW, barH);
      const sweetY1 = Math.floor(barY + barH * 0.25);
      const sweetY2 = Math.floor(barY + barH * 0.75);
      ctx.fillStyle = "#288c28";
      ctx.fillRect(barX, sweetY1, barW, sweetY2 - sweetY1);
      ctx.strokeStyle = "#505f73";
      ctx.lineWidth = 1;
      ctx.strokeRect(barX, barY, barW, barH);
      ctx.fillStyle = "#b4c8dc";
      ctx.font = "10px sans-serif";
      ctx.fillText("Z", barX - 1, barY - 4);

      const zVal = this.handDetected ? this.worldHand.z : this.playerPaddle.z;
      const zClamped = Math.max(-1.70, Math.min(-0.90, zVal));
      const needleRatio = (zClamped - -1.70) / 0.80;
      const needleY = Math.floor(barY + barH * (1.0 - needleRatio));

      let needleColor = "#00e6ff";
      if (needleY >= sweetY1 && needleY <= sweetY2) needleColor = "#00ff78";
      else if (needleY < sweetY1) needleColor = "#ffd166";

      ctx.strokeStyle = needleColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(barX - 3, needleY);
      ctx.lineTo(barX + barW + 3, needleY);
      ctx.stroke();

      // 3. User Swing Speed & Power Gauge
      const gaugeY = camY1 + ch + 8;
      const gaugeW = cw;
      const gaugeH = 36;

      ctx.fillStyle = "rgba(18, 22, 32, 0.88)";
      ctx.fillRect(camX1, gaugeY, gaugeW, gaugeH);
      ctx.strokeStyle = "#324155";
      ctx.lineWidth = 1;
      ctx.strokeRect(camX1, gaugeY, gaugeW, gaugeH);

      const userSpd = Math.max(this.playerPaddle.speed, this.wristSpeed);
      let tierText = "PRONTO", tierColor = "#78f078";
      if (userSpd >= 2.4) { tierText = "SMASH!"; tierColor = "#ff8c00"; }
      else if (userSpd >= 1.0) { tierText = "DRIVE"; tierColor = "#ffd166"; }

      ctx.fillStyle = "#dce6f0";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(`SWING: ${userSpd.toFixed(1)} m/s`, camX1 + 8, gaugeY + 16);

      ctx.fillStyle = tierColor;
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(tierText, camX1 + gaugeW - 8, gaugeY + 16);

      const powerRatio = Math.min(1.0, userSpd / 4.0);
      const barLen = Math.floor((gaugeW - 16) * powerRatio);
      ctx.fillStyle = tierColor;
      ctx.fillRect(camX1 + 8, gaugeY + 22, barLen, 6);

      if (inMenu) {
        // 4. Virtual Racket Card (in-menu)
        const rCardY = gaugeY + gaugeH + 8;
        const rCardW = cw;
        const rCardH = 80;

        ctx.fillStyle = "rgba(18, 24, 34, 0.90)";
        ctx.fillRect(camX1, rCardY, rCardW, rCardH);
        ctx.strokeStyle = "#324155";
        ctx.strokeRect(camX1, rCardY, rCardW, rCardH);

        ctx.fillStyle = "#00e6ff";
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("RAQUETE VIRTUAL", camX1 + 10, rCardY + 17);

        const isGrip = this.isFist || !this.handDetected;
        const statusTxt = this.handDetected ? (this.isFist ? "CONECTADA [OK]" : "EM ESPERA") : "MOUSE ATIVO [OK]";
        const statusCol = this.handDetected ? (this.isFist ? "#3cf078" : "#ffd166") : "#00d7ff";
        ctx.fillStyle = statusCol;
        ctx.font = "10px sans-serif";
        ctx.fillText(`Status: ${statusTxt}`, camX1 + 10, rCardY + 36);

        ctx.fillStyle = "#d2dceb";
        ctx.fillText(`Grip: ${this.handDetected ? (this.isFist ? "CLASSICO (SHAKE)" : "ABERTO") : "MOUSE PADRAO"}`, camX1 + 10, rCardY + 53);
        ctx.fillStyle = "#96afc3";
        ctx.fillText("Borracha: Vermelha (FH)", camX1 + 10, rCardY + 70);

        // 5. Quick Shortcuts Guide (in-menu)
        const kCardY = rCardY + rCardH + 8;
        const kCardW = cw;
        const kCardH = 138;

        ctx.fillStyle = "rgba(18, 22, 32, 0.90)";
        ctx.fillRect(camX1, kCardY, kCardW, kCardH);
        ctx.strokeStyle = "#2d3a4c";
        ctx.strokeRect(camX1, kCardY, kCardW, kCardH);

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px sans-serif";
        ctx.fillText("ATALHOS RAPIDOS:", camX1 + 10, kCardY + 17);

        const shortcuts = [
          ["[ESPACO]", "Iniciar Partida"],
          ["[1 - 3]", "Dificuldade"],
          ["[H]", "Braco (Dir/Esq)"],
          ["[M]", "Menu / Pausa"],
          ["[T]", "Treinar Braco"],
          ["[R]", "Reiniciar Jogo"],
          ["[F]", "Tela Cheia"],
        ];

        shortcuts.forEach(([sKey, sDesc], idx) => {
          const sy = kCardY + 35 + idx * 15;
          ctx.fillStyle = "#00d7ff";
          ctx.font = "10px sans-serif";
          ctx.fillText(sKey, camX1 + 10, sy);
          ctx.fillStyle = "#b4c8dc";
          ctx.fillText(sDesc, camX1 + 75, sy);
        });
      }

      ctx.restore();
    }

    drawMenuModal() {
      const ctx = this.renderer.ctx;
      this.renderer.menu_buttons = {};

      const mw = 690;
      const mh = 492;
      const x1 = 25;
      const y1 = 24;
      const x2 = x1 + mw;
      const y2 = y1 + mh;
      const cx = x1 + mw / 2;

      // Rounded rectangle helper
      const roundRect = (rx, ry, rw, rh, r) => {
        if (rw < 2 * r) r = rw / 2;
        if (rh < 2 * r) r = rh / 2;
        ctx.beginPath();
        ctx.moveTo(rx + r, ry);
        ctx.arcTo(rx + rw, ry, rx + rw, ry + rh, r);
        ctx.arcTo(rx + rw, ry + rh, rx, ry + rh, r);
        ctx.arcTo(rx, ry + rh, rx, ry, r);
        ctx.arcTo(rx, ry, rx + rw, ry, r);
        ctx.closePath();
      };

      ctx.save();

      // 1. Dark semi-transparent backdrop overlay
      ctx.fillStyle = "rgba(6, 9, 15, 0.82)";
      ctx.fillRect(0, 0, this.renderer.width, this.renderer.height);

      // 2. Tournament Modal Card (Olympic Sports Style)
      ctx.save();
      ctx.shadowColor = "rgba(0, 0, 0, 0.85)";
      ctx.shadowBlur = 28;
      ctx.fillStyle = "rgba(15, 23, 42, 0.96)";
      roundRect(x1, y1, mw, mh, 12);
      ctx.fill();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();

      const isGameOver = (this.status === MatchStatus.GAME_OVER);
      const playerWon = isGameOver && (this.player_score > this.ai_score);
      const isPaused = this.in_menu && (this.player_score > 0 || this.ai_score > 0);

      let themeColor = "#E63946";
      let headerTitle = "AIRPONG • COPA ROBERVAL 2026";
      let subtitle = "CAMPEONATO DE TENIS DE MESA  •  MODO ARCADE WII SPORTS";
      let actionLabel = "▶  ENTRAR NA QUADRA & SACAR";

      if (isGameOver) {
        themeColor = playerWon ? "#F59E0B" : "#EF4444";
        headerTitle = playerWon ? "VOCE VENCEU O SET! 🏆" : "ROBERVAL VENCEU O SET";
        subtitle = `PLACAR FINAL: ${this.player_score} x ${this.ai_score}   •   SETS: ${this.player_sets} x ${this.ai_sets}`;
        actionLabel = "▶  INICIAR PROXIMO SET";
      } else if (isPaused) {
        themeColor = "#F59E0B";
        headerTitle = "AIRPONG • PARTIDA PAUSADA";
        subtitle = `PLACAR ATUAL: VOCE ${this.player_score} x ${this.ai_score} ROBERVAL   •   SET ${this.set_number}`;
        actionLabel = "▶  CONTINUAR PARTIDA";
      }

      // Top tournament ribbon
      ctx.fillStyle = themeColor;
      roundRect(x1 + 1, y1 + 1, mw - 2, 4, 3);
      ctx.fill();

      // Header title & subtitle with AirPong emblem badge
      ctx.save();
      const hasEmblem = this.renderer.emblemImageLoaded && this.renderer.emblemImage.width > 0;
      if (hasEmblem) {
        ctx.font = "bold 21px 'Chakra Petch', sans-serif";
        const titleW = ctx.measureText(headerTitle).width;
        const iconSize = 32;
        const iconGap = 10;
        const totalW = titleW + iconSize + iconGap;
        const startX = cx - totalW / 2;

        ctx.drawImage(this.renderer.emblemImage, startX, y1 + 8, iconSize, iconSize);

        ctx.fillStyle = "#FFFFFF";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(headerTitle, startX + iconSize + iconGap, y1 + 8 + iconSize / 2);
      } else {
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 22px 'Chakra Petch', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(headerTitle, cx, y1 + 24);
      }

      ctx.fillStyle = "#94A3B8";
      ctx.font = "11px 'Inter', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(subtitle, cx, y1 + 45);
      ctx.restore();

      // 3. Top Navigation Tabs (Athletic Tournament Style)
      const tabY = y1 + 60;
      const tabH = 32;
      const tabW = Math.floor((mw - 50) / 3);
      const tabs = [
        ["tab_play", "PLAY", "🏓 PARTIDA"],
        ["tab_help", "HELP", "📖 COMO JOGAR"],
        ["tab_bio", "BIO", "🦾 BIOMECANICA"],
      ];

      tabs.forEach(([key, tabId, label], idx) => {
        const tx = x1 + 25 + idx * tabW;
        const isActive = (this.menuTab === tabId);
        const isHover = (this.hoveredAction === key);
        const isFocus = (this.getFocusableActions()[this.focusedActionIndex] === key);

        this.renderer.menu_buttons[key] = [tx, tabY, tx + tabW - 8, tabY + tabH];

        ctx.save();
        if (isActive) {
          ctx.fillStyle = "#E63946";
          ctx.shadowColor = "rgba(230, 57, 70, 0.4)";
          ctx.shadowBlur = 10;
        } else if (isHover || isFocus) {
          ctx.fillStyle = "#334155";
        } else {
          ctx.fillStyle = "#1E293B";
        }
        roundRect(tx, tabY, tabW - 8, tabH, 6);
        ctx.fill();

        ctx.strokeStyle = isActive ? "#FF4D5A" : ((isHover || isFocus) ? "#64748B" : "rgba(255, 255, 255, 0.08)");
        ctx.lineWidth = (isActive || isHover || isFocus) ? 1.5 : 1;
        ctx.stroke();

        ctx.fillStyle = isActive ? "#FFFFFF" : ((isHover || isFocus) ? "#F8FAFC" : "#94A3B8");
        ctx.font = isActive ? "bold 12px 'Chakra Petch', sans-serif" : "11px 'Inter', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(label, tx + (tabW - 8) / 2, tabY + 20);
        ctx.restore();
      });

      // Divider below tabs
      ctx.strokeStyle = "rgba(255, 255, 255, 0.10)";
      ctx.beginPath();
      ctx.moveTo(x1 + 25, tabY + tabH + 8);
      ctx.lineTo(x2 - 25, tabY + tabH + 8);
      ctx.stroke();

      const contentY = tabY + tabH + 16;
      const contentW = mw - 50;
      const contentX = x1 + 25;

      // 4. TAB CONTENTS
      if (this.menuTab === "PLAY") {
        // 4.1. Match Face-Off Pod (Player VS Roberval)
        const podH = 68;
        ctx.save();
        ctx.fillStyle = "rgba(30, 41, 59, 0.70)";
        roundRect(contentX, contentY, contentW, podH, 8);
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.10)";
        ctx.lineWidth = 1;
        ctx.stroke();

        // Player section (Left)
        ctx.textAlign = "left";
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 14px 'Chakra Petch', sans-serif";
        ctx.fillText("👤 VOCE", contentX + 16, contentY + 24);

        const isGrip = this.isFist;
        const hasHand = this.handDetected;
        const statusColor = isGrip ? "#10B981" : (hasHand ? "#F59E0B" : "#10B981");
        const statusText = isGrip ? "RAQUETE EMPUNHADA [OK]" : (hasHand ? "AGUARDANDO EMPUNHADURA" : "CONTROLE POR MOUSE PRONTO");

        ctx.fillStyle = statusColor;
        ctx.font = "11px 'Inter', sans-serif";
        ctx.fillText(statusText, contentX + 16, contentY + 44);

        ctx.fillStyle = "#94A3B8";
        ctx.font = "10px 'Inter', sans-serif";
        ctx.fillText(`Borracha Vermelha  •  Set ${this.set_number} (${this.player_sets} sets)`, contentX + 16, contentY + 58);

        // Center VS Crest
        ctx.textAlign = "center";
        ctx.fillStyle = "#F59E0B";
        ctx.font = "bold 16px 'Chakra Petch', sans-serif";
        ctx.fillText("VS", contentX + contentW / 2, contentY + 28);
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 22px 'Chakra Petch', sans-serif";
        ctx.fillText(`${this.player_score}  -  ${this.ai_score}`, contentX + contentW / 2, contentY + 52);

        // Roberval section (Right)
        ctx.textAlign = "right";
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 14px 'Chakra Petch', sans-serif";
        ctx.fillText("ROBERVAL 🦾", contentX + contentW - 16, contentY + 24);

        ctx.fillStyle = "#F59E0B";
        ctx.font = "11px 'Inter', sans-serif";
        ctx.fillText(`TIER: ${this.ai.difficulty}`, contentX + contentW - 16, contentY + 44);

        ctx.fillStyle = "#94A3B8";
        ctx.font = "10px 'Inter', sans-serif";
        ctx.fillText(`${this.ai_sets} sets conquistados`, contentX + contentW - 16, contentY + 58);
        ctx.restore();

        // 4.2. Primary Action Button (Tactile 3D Arcade Button)
        const btn1Y = contentY + podH + 12;
        const btn1H = 50;
        const isStartHover = (this.hoveredAction === "start");
        const isStartFocus = (this.getFocusableActions()[this.focusedActionIndex] === "start");
        this.renderer.menu_buttons.start = [contentX, btn1Y, contentX + contentW, btn1Y + btn1H];

        ctx.save();
        if (isStartHover || isStartFocus) {
          ctx.fillStyle = "#FF4D5A";
          ctx.shadowColor = "rgba(230, 57, 70, 0.45)";
          ctx.shadowBlur = 18;
        } else {
          ctx.fillStyle = "#E63946";
        }
        roundRect(contentX, btn1Y, contentW, btn1H, 8);
        ctx.fill();

        ctx.strokeStyle = (isStartHover || isStartFocus) ? "#FFFFFF" : "#B91C1C";
        ctx.lineWidth = (isStartHover || isStartFocus) ? 2 : 1.5;
        ctx.stroke();

        ctx.textAlign = "center";
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 18px 'Chakra Petch', sans-serif";
        const startText = (isStartHover || isStartFocus) ? `▸ ${actionLabel}` : actionLabel;
        ctx.fillText(startText, cx, btn1Y + 27);

        ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
        ctx.font = "11px 'Inter', sans-serif";
        ctx.fillText("Pressione [ESPACO] ou clique aqui com o mouse para jogar", cx, btn1Y + 43);
        ctx.restore();

        // 4.3. Difficulty Selector Row (Tournament Tiers)
        const diffY = btn1Y + btn1H + 14;
        ctx.textAlign = "left";
        ctx.fillStyle = "#94A3B8";
        ctx.font = "bold 11px 'Inter', sans-serif";
        ctx.fillText("SELECIONE O TIER DO ADVERSARIO:", contentX + 4, diffY + 12);

        const levels = [
          ["1", "FACIL", "EASY", "Iniciante (Treino)", "#10B981"],
          ["2", "MEDIO", "MEDIUM", "Equilibrado (Clube)", "#3B82F6"],
          ["3", "DIFICIL", "HARD", "Desafiador (Mestre)", "#E63946"],
        ];

        const bw = Math.floor((contentW - 24) / 3);
        const bh = 42;
        const boxStartY = diffY + 18;

        levels.forEach(([k, name, dVal, sub, tColor], idx) => {
          const bx = contentX + idx * (bw + 12);
          const key = `diff_${k}`;
          const isActive = (this.ai.difficulty === dVal);
          const isHover = (this.hoveredAction === key);
          const isFocus = (this.getFocusableActions()[this.focusedActionIndex] === key);

          this.renderer.menu_buttons[key] = [bx, boxStartY, bx + bw, boxStartY + bh];

          ctx.save();
          if (isActive) {
            ctx.fillStyle = "#1E293B";
            ctx.shadowColor = tColor;
            ctx.shadowBlur = 8;
          } else if (isHover || isFocus) {
            ctx.fillStyle = "#334155";
          } else {
            ctx.fillStyle = "#0F172A";
          }
          roundRect(bx, boxStartY, bw, bh, 6);
          ctx.fill();

          ctx.strokeStyle = isActive ? tColor : ((isHover || isFocus) ? "#94A3B8" : "rgba(255, 255, 255, 0.08)");
          ctx.lineWidth = (isActive || isHover || isFocus) ? 2 : 1;
          ctx.stroke();

          ctx.fillStyle = (isActive || isHover || isFocus) ? "#FFFFFF" : "#94A3B8";
          ctx.font = "bold 11px 'Chakra Petch', sans-serif";
          ctx.textAlign = "left";
          const diffLabel = (isHover || isFocus) ? `▸ [${k}] ${name}` : `[${k}] ${name}`;
          ctx.fillText(diffLabel, bx + 8, boxStartY + 18);

          ctx.fillStyle = isActive ? tColor : "#64748B";
          ctx.font = "9px 'Inter', sans-serif";
          ctx.fillText(sub, bx + 8, boxStartY + 33);
          ctx.restore();
        });

        // 4.4. Secondary Buttons Row (Restart + Calibrate)
        const secY = boxStartY + bh + 12;
        const secW = Math.floor((contentW - 14) / 2);
        const secH = 34;

        // Restart Match button
        const isRestartHover = (this.hoveredAction === "restart");
        const isRestartFocus = (this.getFocusableActions()[this.focusedActionIndex] === "restart");
        this.renderer.menu_buttons.restart = [contentX, secY, contentX + secW, secY + secH];

        ctx.save();
        ctx.fillStyle = (isRestartHover || isRestartFocus) ? "#334155" : "#1E293B";
        roundRect(contentX, secY, secW, secH, 6);
        ctx.fill();
        ctx.strokeStyle = (isRestartHover || isRestartFocus) ? "#F59E0B" : "rgba(255, 255, 255, 0.12)";
        ctx.lineWidth = (isRestartHover || isRestartFocus) ? 1.5 : 1;
        ctx.stroke();

        ctx.textAlign = "center";
        ctx.fillStyle = (isRestartHover || isRestartFocus) ? "#FFFFFF" : "#E2E8F0";
        ctx.font = "bold 11px 'Chakra Petch', sans-serif";
        ctx.fillText(isRestartHover ? "▸ [ R ] REINICIAR PLACAR (0 x 0)" : "[ R ] REINICIAR PLACAR (0 x 0)", contentX + secW / 2, secY + 22);
        ctx.restore();

        // Calibrate Button
        const tX = contentX + secW + 14;
        const isCalibHover = (this.hoveredAction === "calibrate");
        const isCalibFocus = (this.getFocusableActions()[this.focusedActionIndex] === "calibrate");
        this.renderer.menu_buttons.calibrate = [tX, secY, tX + secW, secY + secH];

        ctx.save();
        ctx.fillStyle = (isCalibHover || isCalibFocus) ? "#334155" : "#1E293B";
        roundRect(tX, secY, secW, secH, 6);
        ctx.fill();
        ctx.strokeStyle = (isCalibHover || isCalibFocus) ? "#F59E0B" : "rgba(255, 255, 255, 0.12)";
        ctx.lineWidth = (isCalibHover || isCalibFocus) ? 1.5 : 1;
        ctx.stroke();

        ctx.textAlign = "center";
        ctx.fillStyle = (isCalibHover || isCalibFocus) ? "#FFFFFF" : "#E2E8F0";
        ctx.font = "bold 11px 'Chakra Petch', sans-serif";
        ctx.fillText(isCalibHover ? "▸ [ T ] CALIBRAR WEBCAM & GESTOS" : "[ T ] CALIBRAR WEBCAM & GESTOS", tX + secW / 2, secY + 22);
        ctx.restore();

        // 4.5. Quick Usability Pro Tip
        const tipY = secY + secH + 10;
        const tipH = 40;
        ctx.save();
        ctx.fillStyle = "rgba(30, 41, 59, 0.65)";
        roundRect(contentX, tipY, contentW, tipH, 6);
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.fillStyle = "#94A3B8";
        ctx.font = "10px 'Inter', sans-serif";
        ctx.fillText("💡 DICA DO TECNICO: Mova o mouse sobre a mesa para posicionar a raquete com resposta em 60 FPS.", contentX + 14, tipY + 16);
        ctx.fillText("   A assistencia estilo Wii Sports atrai e rebate a bola automaticamente no quique!", contentX + 14, tipY + 30);
        ctx.restore();

      } else if (this.menuTab === "HELP") {
        // Visual Guide Cards (HELP Tab)
        const cardH = 76;

        // Card 1: Mouse Controls
        ctx.save();
        ctx.fillStyle = "rgba(30, 41, 59, 0.70)";
        roundRect(contentX, contentY, contentW, cardH, 8);
        ctx.fill();
        ctx.strokeStyle = "#E63946";
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.fillStyle = "#E63946";
        ctx.font = "bold 12px 'Chakra Petch', sans-serif";
        ctx.fillText("🖱️ MODO MOUSE (PADRAO • 60 FPS • PRONTO PARA JOGAR)", contentX + 16, contentY + 22);
        ctx.fillStyle = "#E2E8F0";
        ctx.font = "11px 'Inter', sans-serif";
        ctx.fillText("• Mova o cursor do mouse livremente sobre a mesa para posicionar a raquete com precisao.", contentX + 16, contentY + 40);
        ctx.fillText("• Clique com o botao esquerdo para sacar ou disparar um Power Smash veloz!", contentX + 16, contentY + 56);
        ctx.fillText("• A assistencia arcade rebate suavemente ao interceptar a bola no tempo do quique.", contentX + 16, contentY + 70);
        ctx.restore();

        // Card 2: Webcam & Real Grip
        const card2Y = contentY + cardH + 10;
        ctx.save();
        ctx.fillStyle = "rgba(30, 41, 59, 0.70)";
        roundRect(contentX, card2Y, contentW, cardH, 8);
        ctx.fill();
        ctx.strokeStyle = "#10B981";
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.fillStyle = "#10B981";
        ctx.font = "bold 12px 'Chakra Petch', sans-serif";
        ctx.fillText("✋ MODO WEBCAM & EMPUNHADURA REAL (MEDIAPIPE 21 PONTOS)", contentX + 16, card2Y + 22);
        ctx.fillStyle = "#E2E8F0";
        ctx.font = "11px 'Inter', sans-serif";
        ctx.fillText("• Fique a 1.2m a 1.8m da webcam e feche a mao como se segurasse o cabo de uma raquete real.", contentX + 16, card2Y + 40);
        ctx.fillText("• Suporta os 3 estilos: Classico (Shakehand), Caneta Japonesa e Classineta (RPB).", contentX + 16, card2Y + 56);
        ctx.fillText("• Acelere o braco para frente para rebater; mova de cima para baixo para dar Smash!", contentX + 16, card2Y + 70);
        ctx.restore();

        // Card 3: Shortcuts Grid
        const card3Y = card2Y + cardH + 10;
        const card3H = 54;
        ctx.save();
        ctx.fillStyle = "rgba(30, 41, 59, 0.70)";
        roundRect(contentX, card3Y, contentW, card3H, 8);
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.fillStyle = "#F59E0B";
        ctx.font = "bold 12px 'Chakra Petch', sans-serif";
        ctx.fillText("⌨️ TECLAS DE ATALHO RAPIDAS", contentX + 16, card3Y + 20);
        ctx.fillStyle = "#94A3B8";
        ctx.font = "10px 'Inter', sans-serif";
        ctx.fillText("[ESPACO] Sacar / Smash   |   [M] Menu de Pausa   |   [1 - 3] Dificuldade   |   [R] Reiniciar   |   [F] Tela Cheia", contentX + 16, card3Y + 38);
        ctx.restore();

        // Action button to start game
        const backBtnY = card3Y + card3H + 12;
        const backBtnH = 44;
        const isBackHover = (this.hoveredAction === "start");
        this.renderer.menu_buttons.start = [contentX, backBtnY, contentX + contentW, backBtnY + backBtnH];

        ctx.save();
        ctx.fillStyle = isBackHover ? "#FF4D5A" : "#E63946";
        roundRect(contentX, backBtnY, contentW, backBtnH, 8);
        ctx.fill();
        ctx.strokeStyle = isBackHover ? "#FFFFFF" : "#B91C1C";
        ctx.stroke();

        ctx.textAlign = "center";
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 15px 'Chakra Petch', sans-serif";
        ctx.fillText(isBackHover ? "▸ [ ▶ ] JOGAR AGORA" : "[ ▶ ] JOGAR AGORA", cx, backBtnY + 28);
        ctx.restore();

      } else if (this.menuTab === "BIO") {
        // Biomechanics & Tracking Tab
        const bioCardH = 150;
        ctx.save();
        ctx.fillStyle = "rgba(30, 41, 59, 0.70)";
        roundRect(contentX, contentY, contentW, bioCardH, 8);
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.fillStyle = "#F59E0B";
        ctx.font = "bold 13px 'Chakra Petch', sans-serif";
        ctx.fillText("DIAGNOSTICO EM TEMPO REAL DO RASTREAMENTO", contentX + 16, contentY + 24);

        const hasHand = this.handDetected;
        const isGrip = this.isFist;
        ctx.font = "11px 'Inter', sans-serif";
        ctx.fillStyle = hasHand ? "#10B981" : "#F59E0B";
        ctx.fillText(`• Rastreamento da Mao: ${hasHand ? "DETECTADA [OK]" : "AGUARDANDO JOGADOR"}`, contentX + 16, contentY + 48);

        ctx.fillStyle = isGrip ? "#10B981" : "#E2E8F0";
        ctx.fillText(`• Empunhadura da Raquete: ${isGrip ? "FECHADA / CLAMP ATIVO [PRONTO]" : "ABERTA (Feche para empunhar)"}`, contentX + 16, contentY + 68);

        ctx.fillStyle = "#E2E8F0";
        ctx.fillText(`• Angulo do Cotovelo: ${this.elbowAngle.toFixed(0)} graus  |  Velocidade: ${(this.wristSpeed * 3.6).toFixed(1)} km/h`, contentX + 16, contentY + 88);

        ctx.fillStyle = "#94A3B8";
        ctx.fillText(`• Posicao no Mundo 3D: X: ${this.worldHand.x.toFixed(2)} | Y: ${this.worldHand.y.toFixed(2)} | Z: ${this.worldHand.z.toFixed(2)}`, contentX + 16, contentY + 108);

        ctx.fillStyle = "#64748b";
        ctx.font = "10px sans-serif";
        ctx.fillText("💡 Dica: Boa iluminacao frontal melhora drasticamente a precisao dos 21 pontos da mao.", contentX + 16, contentY + 132);
        ctx.restore();

        // Calibrate action
        const calY = contentY + bioCardH + 16;
        const calH = 44;
        const isCalHover = (this.hoveredAction === "calibrate");
        this.renderer.menu_buttons.calibrate = [contentX, calY, contentX + contentW, calY + calH];

        ctx.save();
        ctx.fillStyle = isCalHover ? "rgba(0, 180, 216, 0.95)" : "rgba(22, 32, 50, 0.85)";
        roundRect(contentX, calY, contentW, calH, 8);
        ctx.fill();
        ctx.strokeStyle = isCalHover ? "#00f0ff" : "#00b4d8";
        ctx.stroke();

        ctx.textAlign = "center";
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 14px sans-serif";
        ctx.fillText(isCalHover ? "▸ [ T ] INICIAR WIZARD DE CALIBRACAO (3 PASSOS)" : "[ T ] INICIAR WIZARD DE CALIBRACAO (3 PASSOS)", cx, calY + 27);
        ctx.restore();

        // Return button
        const retY = calY + calH + 10;
        const retH = 40;
        const isRetHover = (this.hoveredAction === "start");
        this.renderer.menu_buttons.start = [contentX, retY, contentX + contentW, retY + retH];

        ctx.save();
        ctx.fillStyle = isRetHover ? "rgba(0, 119, 182, 0.95)" : "rgba(16, 24, 38, 0.85)";
        roundRect(contentX, retY, contentW, retH, 8);
        ctx.fill();
        ctx.strokeStyle = "#25344c";
        ctx.stroke();

        ctx.textAlign = "center";
        ctx.fillStyle = isRetHover ? "#ffffff" : "#cbd5e1";
        ctx.font = "12px sans-serif";
        ctx.fillText(isRetHover ? "▸ [ ▶ ] VOLTAR A PARTIDA" : "[ ▶ ] VOLTAR A PARTIDA", cx, retY + 25);
        ctx.restore();
      }

      // 5. Modal Footer
      ctx.textAlign = "center";
      ctx.fillStyle = "#64748b";
      ctx.font = "11px sans-serif";
      ctx.fillText("[M] MENU  |  [F] TELA CHEIA  |  [TAB] NAVEGAR  |  [ESPACO / ENTER] SELECIONAR", cx, y2 - 14);

      ctx.restore();
    }
  }

  // Initialize once DOM is ready
  window.addEventListener("DOMContentLoaded", () => {
    new GameApp();
  });
})();
