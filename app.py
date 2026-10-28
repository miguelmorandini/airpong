"""AI Ping Pong - Interactive 60 FPS Table Tennis for Databricks Apps and Local Play."""

import json
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.databricks.storage import get_storage
from src.databricks.analytics import PingPongAnalytics
from src.engine.game_state import MatchStats, ShotRecord
from src.ui.components import (
    render_header,
    render_kpi_cards,
    render_shot_heatmap,
    render_shot_type_distribution,
    render_speed_history,
)

# 1. Page Configuration
st.set_page_config(
    page_title="AI Ping Pong | Databricks Apps",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. State & Storage Initialization
if "storage" not in st.session_state:
    st.session_state.storage = get_storage()

if "match_stats" not in st.session_state:
    st.session_state.match_stats = MatchStats(
        match_id="live-session",
        start_time=time.time(),
        difficulty="MEDIUM",
    )


# 3. Sidebar Configuration
with st.sidebar:
    st.title("🏓 AI Ping Pong")
    st.caption("Inspirado no Wii Sports & Databricks Apps")

    nav_mode = st.radio(
        "Navegação",
        ["🎮 Jogar Partida (60 FPS)", "📊 Databricks Analytics & Telemetria", "📖 Como Jogar"],
        index=0,
    )

    st.divider()
    st.subheader("Configurações da Partida")
    difficulty = st.selectbox(
        "Dificuldade da IA",
        ["EASY (Fácil / Treino)", "MEDIUM (Equilibrado)", "HARD (Desafiador)", "CHAMPION (Campeão)"],
        index=1,
    )
    diff_code = difficulty.split()[0]

    st.divider()
    st.markdown(
        """
        ### 🎮 Controles Simples:
        - 🖱️ **Mova o mouse** sobre a mesa para controlar a raquete com precisão imediata.
        - 🖱️ **Clique com o mouse** ou pressione **[ESPAÇO]** para sacar e dar *Power Smash*.
        - 🏓 **Assistência Wii Sports**: A raquete atrai e rebate a bola automaticamente quando você se posiciona na área de rebatida.
        """
    )


# HTML5 Canvas 3D Interactive Ping Pong Game Code (60 FPS Native Web Engine)
HTML5_GAME_COMPONENT = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        margin: 0;
        padding: 0;
        background: #0a111e;
        color: #fff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        user-select: none;
        overflow: hidden;
    }}
    #container {{
        position: relative;
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        border: 2px solid #1f3552;
    }}
    canvas {{
        display: block;
        width: 100%;
        height: auto;
        background: linear-gradient(180deg, #99c2ec 0%, #e8f0fe 70%, #d0dbe8 100%);
        cursor: none;
    }}
    #ui-overlay {{
        position: absolute;
        top: 15px;
        left: 20px;
        right: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(14, 25, 45, 0.85);
        padding: 10px 24px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(6px);
    }}
    .score-badge {{
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }}
    .score-player {{ color: #00e5ff; }}
    .score-ai {{ color: #ff5252; }}
    .rally-box {{
        background: #112238;
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 15px;
        color: #ffeb3b;
        font-weight: 700;
        border: 1px solid #2a476f;
    }}
    #banner {{
        position: absolute;
        top: 32%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 40, 120, 0.90);
        color: #fff;
        padding: 12px 30px;
        border-radius: 8px;
        font-size: 26px;
        font-weight: 800;
        border: 3px solid #ffeb3b;
        box-shadow: 0 6px 20px rgba(0,0,0,0.5);
        pointer-events: none;
        transition: opacity 0.3s;
        text-align: center;
    }}
    #bottom-bar {{
        position: absolute;
        bottom: 12px;
        left: 20px;
        right: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        pointer-events: none;
    }}
    .hint-text {{
        font-size: 13px;
        color: rgba(0,0,0,0.7);
        font-weight: 600;
        background: rgba(255,255,255,0.7);
        padding: 4px 12px;
        border-radius: 6px;
    }}
</style>
</head>
<body>

<div id="container">
    <div id="ui-overlay">
        <div class="score-badge score-player">VOCÊ: <span id="p-score">0</span></div>
        <div class="rally-box">RALLY: <span id="rally-count">0</span></div>
        <div class="score-badge score-ai">IA ({diff_code}): <span id="ai-score">0</span></div>
    </div>

    <div id="banner">CLIQUE OU PRESSIONE ESPAÇO PARA SACAR!</div>

    <div id="bottom-bar">
        <span class="hint-text">🖱️ Mova o mouse sobre a mesa para jogar &bull; Clique para Sacar / Smash</span>
        <span class="hint-text" id="speed-indicator">VELOCIDADE: 0 km/h</span>
    </div>

    <canvas id="gameCanvas" width="900" height="560"></canvas>
</div>

<script>
// --- Web Audio Synthesizer (Wii Sports Style Audio Feedback) ---
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playPaddleSound(isSmash) {{
    if (!audioCtx) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = isSmash ? "triangle" : "sine";
    osc.frequency.setValueAtTime(isSmash ? 580 : 420, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(120, audioCtx.currentTime + 0.12);
    gain.gain.setValueAtTime(0.4, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.12);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.12);
}}
function playTableBounceSound() {{
    if (!audioCtx) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(280, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(80, audioCtx.currentTime + 0.08);
    gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.08);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.08);
}}

// --- 3D Table Tennis Simulation Variables ---
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const banner = document.getElementById("banner");
const pScoreEl = document.getElementById("p-score");
const aiScoreEl = document.getElementById("ai-score");
const rallyEl = document.getElementById("rally-count");
const speedEl = document.getElementById("speed-indicator");

// Difficulty Profiles
const DIFF = "{diff_code}";
let aiSpeed = 3.6;
let aiAccuracy = 0.85;
if (DIFF === "EASY") {{ aiSpeed = 2.4; aiAccuracy = 0.70; }}
else if (DIFF === "HARD") {{ aiSpeed = 4.8; aiAccuracy = 0.92; }}
else if (DIFF === "CHAMPION") {{ aiSpeed = 6.2; aiAccuracy = 0.98; }}

// Game State
let playerScore = 0;
let aiScore = 0;
let rallyCount = 0;
let server = "PLAYER";
let status = "SERVING"; // "SERVING", "RALLY", "POINT"
let bannerTimer = 3.0;

// Ball (3D Coordinates in meters: X [-0.76..0.76], Y [0..0.8], Z [-1.37..1.37])
let ball = {{
    x: 0.1, y: 0.28, z: -1.45,
    vx: 0, vy: 0, vz: 0,
    speedKmh: 0,
    inPlay: true,
    lastHitter: "PLAYER",
    trail: []
}};

// Paddles
let playerPaddle = {{ x: 0, y: 0.20, z: -1.65, vx: 0, vy: 0, radius: 0.28 }};
let aiPaddle = {{ x: 0, y: 0.20, z: 1.65, vx: 0, vy: 0, radius: 0.25 }};

// Mouse tracking
let mouseX = 450, mouseY = 400;
let prevMouseX = 450, prevMouseY = 400;

// Camera 3D Perspective Projection
const CAM = {{ x: 0, y: 0.72, z: -2.60, fov: 58 * Math.PI / 180, pitch: 14 * Math.PI / 180 }};
const FOCAL = (canvas.width / 2) / Math.tan(CAM.fov / 2);

function project(x, y, z) {{
    const dx = x - CAM.x;
    const dy = y - CAM.y;
    const dz = z - CAM.z;
    const cosP = Math.cos(CAM.pitch);
    const sinP = Math.sin(CAM.pitch);
    const cy = dy * cosP - dz * sinP;
    const cz = dy * sinP + dz * cosP;
    if (cz <= 0.1) return null;
    return {{
        x: canvas.width / 2 + (FOCAL * dx) / cz,
        y: canvas.height / 2 - (FOCAL * cy) / cz,
        z: cz
    }};
}}

// Serve Ball
function serve() {{
    status = "RALLY";
    ball.inPlay = true;
    ball.trail = [];
    if (server === "PLAYER") {{
        ball.x = playerPaddle.x * 0.5;
        ball.y = 0.30;
        ball.z = -1.45;
        ball.vx = (Math.random() - 0.5) * 0.4;
        ball.vy = 1.6;
        ball.vz = 2.8; // Gentle, predictable launch
        ball.lastHitter = "PLAYER";
    }} else {{
        ball.x = aiPaddle.x * 0.5;
        ball.y = 0.30;
        ball.z = 1.45;
        ball.vx = (Math.random() - 0.5) * 0.4;
        ball.vy = 1.6;
        ball.vz = -2.8;
        ball.lastHitter = "AI";
    }}
    showBanner("PLAY!", 1.0);
    playPaddleSound(false);
}}

function showBanner(text, dur = 2.0) {{
    banner.innerText = text;
    banner.style.display = "block";
    banner.style.opacity = "1";
    bannerTimer = dur;
}}

// Input Listeners
canvas.addEventListener("mousemove", (e) => {{
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    mouseX = (e.clientX - rect.left) * scaleX;
    mouseY = (e.clientY - rect.top) * scaleY;
}});

window.addEventListener("mousedown", () => {{
    if (status === "SERVING" || status === "READY") {{
        serve();
    }} else if (status === "RALLY") {{
        // Manual smash trigger if ball is nearby
        if (Math.abs(ball.z - playerPaddle.z) < 0.6) {{
            hitPlayerBall(true);
        }}
    }}
}});

window.addEventListener("keydown", (e) => {{
    if (e.code === "Space") {{
        if (status === "SERVING" || status === "READY") serve();
    }}
}});

// Player Hit with Generous Wii Sports Arcade Assistance
function hitPlayerBall(isSmash = false) {{
    if (ball.vz < 0 && ball.z < -1.10) {{
        ball.lastHitter = "PLAYER";
        rallyCount++;
        rallyEl.innerText = rallyCount;

        // Ball rebound
        const forwardSpeed = isSmash ? 5.2 : 3.4;
        ball.vz = forwardSpeed;
        ball.vy = isSmash ? 1.2 : 1.9; // High predictable arc
        ball.vx = (ball.x - playerPaddle.x) * 2.2 + (Math.random() - 0.5) * 0.3;

        ball.speedKmh = Math.sqrt(ball.vx**2 + ball.vy**2 + ball.vz**2) * 3.6;
        speedEl.innerText = "VELOCIDADE: " + ball.speedKmh.toFixed(1) + " km/h";

        playPaddleSound(isSmash);
        showBanner(isSmash ? "POWER SMASH!" : "NICE SHOT!", 0.8);
    }}
}}

// Award Point & Check Set Win
function awardPoint(winner, reason) {{
    status = "POINT";
    if (winner === "PLAYER") {{
        playerScore++;
        pScoreEl.innerText = playerScore;
        showBanner("PONTO SEU! " + reason, 2.0);
    }} else {{
        aiScore++;
        aiScoreEl.innerText = aiScore;
        showBanner("PONTO DA IA! " + reason, 2.0);
    }}

    // Check game over (11 points, lead by 2)
    if ((playerScore >= 11 || aiScore >= 11) && Math.abs(playerScore - aiScore) >= 2) {{
        const champ = playerScore > aiScore ? "VOCÊ VENCEU A PARTIDA! 🏆" : "A IA VENCEU A PARTIDA!";
        showBanner(champ, 4.0);
        status = "GAME_OVER";
        return;
    }}

    // Next server
    const totalPts = playerScore + aiScore;
    server = (Math.floor(totalPts / 2) % 2 === 0) ? "PLAYER" : "AI";

    setTimeout(() => {{
        status = "SERVING";
        if (server === "AI") {{
            setTimeout(serve, 1000);
        }} else {{
            showBanner("SEU SAQUE! CLIQUE PARA SACAR", 3.0);
        }}
    }}, 1800);
}}

// --- Main 60 FPS Game Loop ---
let lastTime = performance.now();

function update(dt) {{
    if (bannerTimer > 0) {{
        bannerTimer -= dt;
        if (bannerTimer <= 0) banner.style.display = "none";
    }}

    // Map mouse to paddle world (X: -0.90..0.90, Y: 0.05..0.65)
    const targetX = ((mouseX / canvas.width) - 0.5) * 1.85;
    const targetY = Math.max(0.05, Math.min(0.65, (1.0 - (mouseY / canvas.height)) * 0.75));

    playerPaddle.vx = (targetX - playerPaddle.x) / dt;
    playerPaddle.vy = (targetY - playerPaddle.y) / dt;
    playerPaddle.x = targetX;
    playerPaddle.y = targetY;

    if (status === "RALLY") {{
        // Ball Physics (Gravity + Step)
        ball.vy -= 9.81 * dt;
        ball.x += ball.vx * dt;
        ball.y += ball.vy * dt;
        ball.z += ball.vz * dt;

        // Trail
        ball.trail.push({{ x: ball.x, y: ball.y, z: ball.z }});
        if (ball.trail.length > 10) ball.trail.shift();

        // AI Paddle Movement
        if (ball.vz > 0) {{
            const targetAiX = ball.x * aiAccuracy;
            const targetAiY = Math.max(0.12, ball.y * 0.85);
            aiPaddle.x += (targetAiX - aiPaddle.x) * Math.min(1.0, aiSpeed * dt * 2.5);
            aiPaddle.y += (targetAiY - aiPaddle.y) * Math.min(1.0, aiSpeed * dt * 2.5);
        }} else {{
            aiPaddle.x += (0 - aiPaddle.x) * 0.05;
            aiPaddle.y += (0.22 - aiPaddle.y) * 0.05;
        }}

        // AI Strike Check
        if (ball.vz > 0 && ball.z >= 1.35) {{
            const distAi = Math.hypot(ball.x - aiPaddle.x, ball.y - aiPaddle.y);
            if (distAi < 0.42) {{
                ball.lastHitter = "AI";
                rallyCount++;
                rallyEl.innerText = rallyCount;
                ball.vz = -3.2;
                ball.vy = 1.8;
                ball.vx = (Math.random() - 0.5) * 1.2;
                playPaddleSound(false);
            }}
        }}

        // Player Assisted Hit Zone (Wii Sports Style)
        if (ball.vz < 0 && ball.z <= -1.25 && ball.z >= -1.75) {{
            const distP = Math.hypot(ball.x - playerPaddle.x, ball.y - playerPaddle.y);
            // Generous 0.50m reach: if you move toward the ball, you hit it!
            if (distP <= 0.50) {{
                hitPlayerBall(false);
            }}
        }}

        // Table Bounce
        if (ball.y <= 0.02) {{
            const onTable = (Math.abs(ball.x) <= 0.7625 && Math.abs(ball.z) <= 1.37);
            if (onTable) {{
                ball.y = 0.02;
                ball.vy = Math.max(1.2, -ball.vy * 0.88);
                playTableBounceSound();
            }}
        }}

        // Out of Bounds Detection
        if (ball.y < -0.3 || Math.abs(ball.z) > 2.2 || Math.abs(ball.x) > 1.5) {{
            if (ball.lastHitter === "PLAYER") {{
                awardPoint("PLAYER", "BOLA FORA!");
            }} else {{
                awardPoint("AI", "BOLA FORA!");
            }}
        }}
    }}
}}

// --- 3D Scene Rendering ---
function render() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Gymnasium Floor
    const fl1 = project(-3.2, -0.76, -1.8);
    const fl2 = project(3.2, -0.76, -1.8);
    const fl3 = project(3.2, -0.76, 3.2);
    const fl4 = project(-3.2, -0.76, 3.2);
    if (fl1 && fl2 && fl3 && fl4) {{
        ctx.fillStyle = "#c5d7ed";
        ctx.beginPath();
        ctx.moveTo(fl1.x, fl1.y);
        ctx.lineTo(fl2.x, fl2.y);
        ctx.lineTo(fl3.x, fl3.y);
        ctx.lineTo(fl4.x, fl4.y);
        ctx.closePath();
        ctx.fill();
    }}

    // 2. 3D ITTF Table Top
    const w2 = 0.7625, l2 = 1.37;
    const c1 = project(-w2, 0, -l2); // Player left
    const c2 = project(w2, 0, -l2);  // Player right
    const c3 = project(w2, 0, l2);   // AI right
    const c4 = project(-w2, 0, l2);  // AI left

    if (c1 && c2 && c3 && c4) {{
        // Blue Table Quad
        ctx.fillStyle = "#105ca8";
        ctx.beginPath();
        ctx.moveTo(c1.x, c1.y);
        ctx.lineTo(c2.x, c2.y);
        ctx.lineTo(c3.x, c3.y);
        ctx.lineTo(c4.x, c4.y);
        ctx.closePath();
        ctx.fill();

        // White border line
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.stroke();

        // White Center Dividing Line
        const cm1 = project(0, 0, -l2);
        const cm2 = project(0, 0, l2);
        if (cm1 && cm2) {{
            ctx.beginPath();
            ctx.moveTo(cm1.x, cm1.y);
            ctx.lineTo(cm2.x, cm2.y);
            ctx.lineWidth = 2;
            ctx.stroke();
        }}
    }}

    // 3. 3D Net
    const netL = project(-0.90, 0.1525, 0);
    const netR = project(0.90, 0.1525, 0);
    const netBL = project(-0.90, 0, 0);
    const netBR = project(0.90, 0, 0);
    if (netL && netR && netBL && netBR) {{
        ctx.fillStyle = "rgba(40, 45, 55, 0.75)";
        ctx.beginPath();
        ctx.moveTo(netBL.x, netBL.y);
        ctx.lineTo(netBR.x, netBR.y);
        ctx.lineTo(netR.x, netR.y);
        ctx.lineTo(netL.x, netL.y);
        ctx.closePath();
        ctx.fill();

        // Net Top White Cord
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(netL.x, netL.y);
        ctx.lineTo(netR.x, netR.y);
        ctx.stroke();
    }}

    // 4. Ball Shadow on Table
    const bShadow = project(ball.x, 0, ball.z);
    if (bShadow) {{
        const r = Math.max(3, 16 / bShadow.z);
        ctx.fillStyle = "rgba(20, 30, 45, 0.4)";
        ctx.beginPath();
        ctx.ellipse(bShadow.x, bShadow.y, r, r * 0.45, 0, 0, Math.PI * 2);
        ctx.fill();
    }}

    // 5. Landing Bullseye Reticle (Anticipation Target)
    if (ball.inPlay && ball.vz < 0 && ball.z > -1.2) {{
        const targetProj = project(ball.x + ball.vx * 0.3, 0, ball.z + ball.vz * 0.3);
        if (targetProj) {{
            ctx.strokeStyle = "#ffea00";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(targetProj.x, targetProj.y, Math.max(6, 25 / targetProj.z), 0, Math.PI * 2);
            ctx.stroke();
        }}
    }}

    // 6. AI Paddle (Black rubber)
    const aiProj = project(aiPaddle.x, aiPaddle.y, aiPaddle.z);
    if (aiProj) {{
        const r = Math.max(8, 48 / aiProj.z);
        ctx.fillStyle = "#222222";
        ctx.beginPath();
        ctx.arc(aiProj.x, aiProj.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#cccccc";
        ctx.lineWidth = 2;
        ctx.stroke();
    }}

    // 7. 3D Ball (Vibrant ITTF 3-Star Orange)
    const bProj = project(ball.x, ball.y, ball.z);
    if (bProj) {{
        const r = Math.max(4, 26 / bProj.z);
        ctx.fillStyle = "#ff9100";
        ctx.beginPath();
        ctx.arc(bProj.x, bProj.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#d86b00";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Highlight
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(bProj.x - r * 0.3, bProj.y - r * 0.3, r * 0.35, 0, Math.PI * 2);
        ctx.fill();
    }}

    // 8. Player Paddle (Red rubber with aim guide)
    const pProj = project(playerPaddle.x, playerPaddle.y, playerPaddle.z);
    if (pProj) {{
        const r = Math.max(12, 65 / pProj.z);

        // Guideline to ball when near
        if (bProj && ball.z < -0.5) {{
            const dist = Math.hypot(ball.x - playerPaddle.x, ball.y - playerPaddle.y);
            const inReach = dist <= 0.50;
            ctx.strokeStyle = inReach ? "#00e676" : "#ff1744";
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(pProj.x, pProj.y);
            ctx.lineTo(bProj.x, bProj.y);
            ctx.stroke();
            ctx.setLineDash([]);

            if (inReach && ball.z < -1.1) {{
                ctx.fillStyle = "#00e676";
                ctx.font = "bold 15px sans-serif";
                ctx.fillText("REBATA AGORA!", pProj.x - 55, pProj.y - r - 12);
            }}
        }}

        // Handle
        ctx.strokeStyle = "#8d6e63";
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(pProj.x, pProj.y + r * 0.4);
        ctx.lineTo(pProj.x, pProj.y + r * 1.2);
        ctx.stroke();

        // Blade
        ctx.fillStyle = "#d50000";
        ctx.beginPath();
        ctx.arc(pProj.x, pProj.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.stroke();
    }}
}}

function loop(timestamp) {{
    const dt = Math.min(0.04, (timestamp - lastTime) / 1000);
    lastTime = timestamp;

    update(dt);
    render();
    requestAnimationFrame(loop);
}}

requestAnimationFrame(loop);
</script>
</body>
</html>
"""


# 4. View: Jogo Interativo (60 FPS Native Web)
if nav_mode == "🎮 Jogar Partida (60 FPS)":
    render_header()

    # Match KPI metrics
    render_kpi_cards(
        st.session_state.match_stats,
        st.session_state.match_stats.player_score,
        st.session_state.match_stats.ai_score,
        "VOCÊ",
    )

    st.markdown("### 🏓 Quadra 3D Interativa (60 FPS em Tempo Real)")
    st.info("💡 **Como Jogar**: Basta mover o mouse sobre a mesa para controlar a raquete com resposta instantânea! Clique na tela para sacar ou dar um smash. A assistência no estilo Wii Sports garante rebatidas intuitivas e divertidas!")

    # Render 60 FPS HTML5 Canvas Component
    components.html(HTML5_GAME_COMPONENT, height=620, scrolling=False)

    st.divider()

    col_desk1, col_desk2 = st.columns([6, 4])
    with col_desk1:
        st.markdown(
            """
            #### 📹 Prefere jogar com a Webcam pelo corpo/mão?
            Você pode executar o modo **Desktop 60 FPS com OpenCV & MediaPipe** direto no seu terminal:
            ```bash
            python main.py
            ```
            Ele abrirá uma janela nativa em tela cheia com rastreamento contínuo dos 21 pontos da mão pela webcam, além de suporte híbrido ao mouse!
            """
        )
    with col_desk2:
        st.markdown(
            """
            #### 💾 Salvar Estatísticas
            Ao final de cada partida, os dados são automaticamente estruturados no schema **Delta Lake** do Databricks (`ping_pong_matches` e `ping_pong_shots`).
            """
        )
        if st.button("💾 Persistir Partida de Demonstração nas Tabelas Delta", use_container_width=True):
            # Create sample completed match for demo
            sample_stats = MatchStats(
                match_id=f"match-{int(time.time())}",
                start_time=time.time() - 120,
                end_time=time.time(),
                player_score=11,
                ai_score=8,
                winner="PLAYER",
                difficulty=diff_code,
                total_rallies=14,
                longest_rally=8,
                avg_speed_kmh=26.4,
                player_smashes=4,
            )
            # Add sample shots
            sample_stats.shots = [
                ShotRecord("s1", sample_stats.match_id, 1, "PLAYER", "SERVE", 18.5, "NONE", 0.1, 0.7, "IN_PLAY"),
                ShotRecord("s2", sample_stats.match_id, 2, "AI", "DRIVE", 22.0, "NONE", -0.2, -0.6, "IN_PLAY"),
                ShotRecord("s3", sample_stats.match_id, 3, "PLAYER", "SMASH", 38.2, "NONE", 0.4, 1.1, "WINNER"),
            ]
            st.session_state.storage.save_match(sample_stats)
            st.success("✅ Partida registrada nas tabelas Delta Lake com sucesso! Acesse a aba de Analytics para ver o heatmap.")


# 5. View: Databricks Analytics & Telemetria
elif nav_mode == "📊 Databricks Analytics & Telemetria":
    render_header()
    storage = st.session_state.storage

    st.subheader("🏆 Hub de Torneios e Análise Estatística (Databricks)")
    agg = storage.get_aggregated_stats()

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total de Partidas", agg["total_matches"])
    with m2:
        st.metric("Vitórias do Jogador", agg["player_wins"])
    with m3:
        st.metric("Taxa de Vitória", f"{agg['win_rate_pct']}%")
    with m4:
        st.metric("Recorde de Rally", f"{agg['record_rally']} trocas")
    with m5:
        st.metric("Total de Smashes", agg["total_smashes"])

    st.divider()

    matches = storage.get_matches(limit=50)
    if matches:
        df_matches = pd.DataFrame(matches)
        st.markdown("### 📋 Histórico de Partidas (Tabela Delta: `ping_pong_matches`)")
        st.dataframe(
            df_matches[[
                "match_id", "player_score", "ai_score", "winner",
                "difficulty", "total_rallies", "longest_rally", "avg_speed_kmh", "player_smashes"
            ]],
            use_container_width=True,
        )

        selected_match_id = st.selectbox(
            "Selecione uma partida para análise aprofundada de rebatidas",
            df_matches["match_id"].tolist(),
        )

        if selected_match_id:
            shots = storage.get_shots_by_match(selected_match_id)
            if shots:
                breakdown = PingPongAnalytics.compute_match_breakdown(shots)
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Total de Golpes", breakdown["total_shots"])
                with b2:
                    st.metric("Erros Não-Forçados", breakdown["player_unforced_errors"])
                with b3:
                    st.metric("Winners / Rebatidas Vencedoras", breakdown["player_winners"])
                with b4:
                    st.metric("Velocidade Máxima", f"{breakdown['max_speed']} km/h")

                st.markdown("#### Mapa de Calor de Impacto da Bola na Mesa (2D Heatmap)")
                shot_objs = [
                    ShotRecord(
                        shot_id=s["shot_id"],
                        match_id=s["match_id"],
                        shot_number=s["shot_number"],
                        hitter=s["hitter"],
                        shot_type=s["shot_type"],
                        speed_kmh=s["speed_kmh"],
                        spin=s["spin"],
                        landing_x=s["landing_x"],
                        landing_z=s["landing_z"],
                        result=s["result"],
                        timestamp=s["timestamp"],
                    )
                    for s in shots
                ]
                render_shot_heatmap(shot_objs)
    else:
        st.info("Nenhuma partida persistida ainda. Jogue uma partida ou clique no botão de salvar para gerar os relatórios analíticos.")


# 6. View: Como Jogar
elif nav_mode == "📖 Como Jogar":
    render_header()
    st.subheader("📖 Guia de Jogo e Regras ITTF")
    st.markdown(
        """
        ### 🏓 Como Rebater a Bola:
        1. **Posicionamento**: O cursor do mouse controla a raquete vermelha diretamente na tela em 3D.
        2. **Zona de Rebatida & Linha Verde**:
           - Quando a bola vem em sua direção, observe o alvo amarelo projetado na mesa onde a bola vai quicar.
           - Após o quique na sua mesa, a linha tracejada entre sua raquete e a bola ficará **verde**.
           - Basta mover sua raquete para perto da bola — com a assistência arcade do Wii Sports, ela rebaterá automaticamente!
        3. **Power Smash**:
           - Clique com o botão esquerdo do mouse no momento do impacto para desferir um smash potente para o fundo da mesa adversária.
        4. **Saque**:
           - Clique na mesa ou pressione **[ESPAÇO]** para lançar o saque.
        """
    )
