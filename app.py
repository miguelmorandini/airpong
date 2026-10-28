"""AI Ping Pong - Main Streamlit Application for Databricks Apps and Local Play."""

import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CameraConfig, DisplayConfig
from src.engine.game_state import GameStateMachine, MatchStatus
from src.engine.paddle import Paddle
from src.ui.renderer_3d import PerspectiveRenderer3D
from src.ui.components import (
    render_header,
    render_kpi_cards,
    render_shot_heatmap,
    render_shot_type_distribution,
    render_speed_history,
)
from src.vision.camera import SyntheticCameraSource, WebcamSource
from src.vision.tracker import HandTracker
from src.databricks.storage import get_storage
from src.databricks.analytics import PingPongAnalytics


# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="AI Ping Pong | Databricks Apps",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. State Initialization
if "game_state" not in st.session_state:
    st.session_state.game_state = GameStateMachine(difficulty="MEDIUM")

if "renderer" not in st.session_state:
    st.session_state.renderer = PerspectiveRenderer3D()

if "tracker" not in st.session_state:
    st.session_state.tracker = HandTracker()

if "storage" not in st.session_state:
    st.session_state.storage = get_storage()

if "camera_source" not in st.session_state:
    st.session_state.camera_source = None

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "last_frame_time" not in st.session_state:
    st.session_state.last_frame_time = time.time()


# 3. Sidebar Configuration
with st.sidebar:
    st.image("https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg", width=50)
    st.title("🏓 AI Ping Pong")
    st.caption("Controle por Visão Computacional & Databricks Apps")

    nav_mode = st.radio(
        "Navegação",
        ["🎮 Jogar Partida", "📊 Databricks Analytics & Telemetria", "⚙️ Configurações"],
        index=0,
    )

    st.divider()

    st.subheader("Configurações da Partida")
    difficulty = st.selectbox(
        "Dificuldade da IA",
        ["EASY", "MEDIUM", "HARD", "CHAMPION"],
        index=1,
    )
    if difficulty != st.session_state.game_state.difficulty:
        st.session_state.game_state.ai.set_difficulty(difficulty)
        st.session_state.game_state.difficulty = difficulty

    control_mode = st.selectbox(
        "Método de Controle",
        ["Controle Virtual Responsivo (Nuvem / Browser)", "Câmera Virtual (Simulação)", "Webcam Local (OpenCV)"],
        index=0,
    )

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Novo Jogo", use_container_width=True):
            # Save prior match stats if played
            if st.session_state.game_state.stats.shots:
                st.session_state.storage.save_match(st.session_state.game_state.stats)
            st.session_state.game_state.start_new_match(difficulty)
            st.success("Nova partida iniciada!")

    with col_btn2:
        if st.button("🎾 Sacar Bola", use_container_width=True, type="primary"):
            st.session_state.game_state.trigger_serve()

    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #888;">
            <b>Regras Oficiais ITTF:</b><br>
            &bull; Sets de 11 pontos (mínimo 2 de diferença).<br>
            &bull; Alternância de saque a cada 2 pontos.<br>
            &bull; Deuce a partir de 10-10.<br>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 4. View: Jogo Interativo
if nav_mode == "🎮 Jogar Partida":
    render_header()

    game = st.session_state.game_state
    renderer = st.session_state.renderer
    tracker = st.session_state.tracker

    # KPI Top Bar
    render_kpi_cards(game.stats, game.player_score, game.ai_score, game.server)

    # Main Game Area
    col_game, col_cam = st.columns([7, 3])

    with col_cam:
        st.markdown("### 📷 Controle da Raquete")

        if control_mode == "Controle Virtual Responsivo (Nuvem / Browser)":
            st.info("💡 Modo ideal para Databricks Apps: controle a raquete e execute movimentos fluidos sem depender de portas UDP restritas.")

            paddle_x = st.slider("Posição Lateral (X)", min_value=-0.85, max_value=0.85, value=float(game.player_paddle.x), step=0.05)
            paddle_y = st.slider("Altura da Raquete (Y)", min_value=0.05, max_value=0.55, value=float(game.player_paddle.y), step=0.05)

            col_sw1, col_sw2 = st.columns(2)
            with col_sw1:
                hit_forehand = st.button("💥 Rebatida Forehand", use_container_width=True)
            with col_sw2:
                hit_smash = st.button("⚡ Power Smash!", use_container_width=True)

            # Update paddle kinematics
            now = time.time()
            dt = max(1e-4, now - st.session_state.last_frame_time)
            st.session_state.last_frame_time = now

            game.player_paddle.update_position(paddle_x, paddle_y, -1.65, dt)
            if hit_forehand:
                game.player_paddle.vx = 4.5
                game.player_paddle.vy = 1.0
                game.player_paddle.vz = 5.0
            elif hit_smash:
                game.player_paddle.vx = 0.5
                game.player_paddle.vy = -2.0
                game.player_paddle.vz = 9.0

            # Step game loop
            game.update(dt=0.033)

        elif control_mode == "Câmera Virtual (Simulação)":
            st.info("🤖 Mão virtual animada simulando rebatidas em tempo real para testes de visão.")
            if st.session_state.camera_source is None or not isinstance(st.session_state.camera_source, SyntheticCameraSource):
                st.session_state.camera_source = SyntheticCameraSource(width=320, height=240)

            ret, frame = st.session_state.camera_source.read()
            if ret and frame is not None:
                track_res = tracker.process_frame(frame)
                dt = 0.033
                game.player_paddle.update_position(track_res.world_x, track_res.world_y, track_res.world_z, dt)
                game.update(dt=dt)
                st.image(cv2.cvtColor(track_res.annotated_frame, cv2.COLOR_BGR2RGB), caption="Rastreamento Virtual MediaPipe", use_container_width=True)

        elif control_mode == "Webcam Local (OpenCV)":
            st.info("📹 Captura direta via webcam local.")
            if st.session_state.camera_source is None or not isinstance(st.session_state.camera_source, WebcamSource):
                st.session_state.camera_source = WebcamSource(device_index=0, width=320, height=240)

            ret, frame = st.session_state.camera_source.read()
            if ret and frame is not None:
                track_res = tracker.process_frame(frame)
                dt = 0.033
                game.player_paddle.update_position(track_res.world_x, track_res.world_y, track_res.world_z, dt)
                game.update(dt=dt)
                st.image(cv2.cvtColor(track_res.annotated_frame, cv2.COLOR_BGR2RGB), caption="Webcam Local + MediaPipe", use_container_width=True)
            else:
                st.warning("Webcam não detectada. Alternando para o controle virtual.")
                game.update(dt=0.033)

        # Quick match action buttons
        st.write("")
        if game.status == MatchStatus.GAME_OVER:
            st.warning(f"🏆 Fim de Jogo! Vencedor: {game.stats.winner}")
            if st.button("Salvar Partida no Databricks", type="primary", use_container_width=True):
                st.session_state.storage.save_match(game.stats)
                st.success("Partida registrada nas tabelas Delta Lake!")

    with col_game:
        # Render 3D Perspective Table Tennis Frame
        scene_frame = renderer.render_frame(game)
        rgb_scene = cv2.cvtColor(scene_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_scene, caption="Visão 3D Inspirada no Wii Sports", use_container_width=True)

        col_act1, col_act2, col_act3 = st.columns([3, 3, 4])
        with col_act1:
            if st.button("▶️ Próximo Frame (Play Loop)", use_container_width=True):
                st.rerun()
        with col_act2:
            auto_play = st.checkbox("Loop Automático Contínuo", value=False)
            if auto_play:
                time.sleep(0.04)
                st.rerun()
        with col_act3:
            st.caption(f"Status: **{game.status.value}** | Rebatidas nesta troca: **{game.current_rally_count}**")

    # Realtime analytics row below game
    st.divider()
    st.subheader("📈 Telemetria da Partida em Andamento")
    tab_map, tab_dist, tab_speed = st.tabs(["🗺️ Heatmap da Mesa", "📊 Tipos de Golpe", "⚡ Velocidade"])
    with tab_map:
        render_shot_heatmap(game.stats.shots)
    with tab_dist:
        render_shot_type_distribution(game.stats.shots)
    with tab_speed:
        render_speed_history(game.stats.shots)


# 5. View: Databricks Analytics & Telemetria
elif nav_mode == "📊 Databricks Analytics & Telemetria":
    render_header()
    storage = st.session_state.storage

    st.subheader("🏆 Hub de Torneios e Análise Estatística (Databricks)")

    # Lifetime aggregates
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

    # Matches table
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

        # Match selection for deep-dive
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

                st.markdown("#### Distribuição de Rebatidas na Mesa")
                from src.engine.game_state import ShotRecord
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
        st.info("Nenhuma partida finalizada persistida ainda. Jogue uma partida e salve os dados para visualizar a análise completa.")


# 6. View: Configurações & Arquitetura
elif nav_mode == "⚙️ Configurações":
    render_header()
    st.subheader("⚙️ Configurações do Sistema e Arquitetura")

    st.markdown(
        """
        ### Arquitetura do AI Ping Pong

        O projeto opera sob arquitetura desacoplada em 4 camadas principais:
        1. **Motor de Física 3D & Regras ITTF (`src/engine`)**: Simulação contínua com gravidade, arrasto aerodinâmico, efeito Magnus (spin), colisões com mesa e rede, além de máquina de estados oficial com sets de 11 pontos e deuce.
        2. **Visão Computacional (`src/vision`)**: Rastreamento de landmarks da mão com MediaPipe Hands em 60 FPS, estimativa de profundidade e aceleração para golpes (forehand, backhand, smash).
        3. **Interface e Renderizador 3D (`src/ui`)**: Projeção de perspectiva com estética inspirada no clássico Wii Sports, sombras realistas, efeitos visuais de rebatida e HUD interativo.
        4. **Databricks Apps & Unity Catalog (`src/databricks`)**: Persistência de telemetria das partidas em tabelas Delta Lake com fallback local resiliente.

        ### Implantação no Databricks Apps
        O manifesto `app.yaml` já está configurado na raiz do repositório:
        ```yaml
        command:
          - "streamlit"
          - "run"
          - "app.py"
          - "--server.port"
          - "8501"
          - "--server.address"
          - "0.0.0.0"
        ```
        """
    )
