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
    inject_custom_theme,
    render_header,
    render_kpi_cards,
    render_shot_heatmap,
    render_shot_type_distribution,
    render_speed_history,
    render_dataset_agent_panel,
)

# 1. Page Configuration
st.set_page_config(
    page_title="AI Ping Pong | Databricks Apps",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject modern UI/UX design tokens and dark stadium theme
inject_custom_theme()

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
    st.image("assets/airpong_logo.png", use_container_width=True)
    st.caption("AirPong: Inspirado no Wii Sports & Databricks Apps")

    nav_mode = st.radio(
        "Navegação",
        [
            "🎮 Jogar Partida (60 FPS)",
            "📊 Databricks Analytics & Telemetria",
            "🤖 Dataset & Treino de Movimentos",
            "📖 Como Jogar",
        ],
        index=0,
    )

    st.divider()
    st.subheader("Configurações da Partida")
    difficulty = st.selectbox(
        "Dificuldade do Roberval",
        ["EASY (Fácil - Iniciante)", "MEDIUM (Equilibrado)", "HARD (Desafiador)"],
        index=0,
    )
    diff_code = difficulty.split()[0]

    st.divider()
    st.markdown(
        """
        ### 🎮 Controles Arcade:
        - 🖱️ **Mova o mouse** sobre a mesa para mover a raquete com resposta em 60 FPS.
        - 🖱️ **Clique com o mouse** ou pressione **[ESPAÇO]** para sacar e dar *Power Smash*.
        - 🏓 **Assistência Wii Sports**: A raquete atrai e rebate a bola automaticamente quando você se posiciona no tempo do quique!
        """
    )


# Unified 1:1 HTML5 Ping Pong Game Component (identical to Desktop OpenCV version)
@st.cache_data
def get_web_game_component() -> str:
    """Build unified 1:1 HTML5 Ping Pong game component with embedded arena texture."""
    import base64
    from pathlib import Path

    docs_dir = Path(__file__).parent / "docs"
    css_path = docs_dir / "style.css"
    js_path = docs_dir / "game.js"
    bg_path = Path(__file__).parent / "assets" / "background_arena.png"

    css_content = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    js_content = js_path.read_text(encoding="utf-8") if js_path.exists() else ""

    bg_data_uri = ""
    if bg_path.exists():
        bg_b64 = base64.b64encode(bg_path.read_bytes()).decode("ascii")
        bg_data_uri = f"data:image/png;base64,{bg_b64}"

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>AI Ping Pong - Wii Sports Mode</title>
  <style>
    {css_content}
  </style>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:ital,wght@0,600;0,700;1,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin />
  <script defer src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
</head>
<body>
  <div id="game-container">
    <canvas id="game-canvas" width="1024" height="640"></canvas>
    <video id="webcam" playsinline style="display: none;"></video>
  </div>
  <script>
    window.BG_ARENA_IMAGE_DATA = "{bg_data_uri}";
  </script>
  <script>
    {js_content}
  </script>
</body>
</html>"""


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
    components.html(get_web_game_component(), height=720, scrolling=False)

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
        if st.button("💾 Persistir Partida de Demonstração nas Tabelas Delta", width="stretch"):
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
            width="stretch",
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


# 6. View: Dataset Agent & Treinamento de Movimentos
elif nav_mode == "🤖 Dataset & Treino de Movimentos":
    render_header()
    import os
    import json
    from src.dataset_agent.agent import TableTennisDatasetAgent

    meta_path = os.path.join("src", "vision", "models", "stroke_model_meta.json")
    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = None

    render_dataset_agent_panel(meta)

    st.divider()
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("🎯 Ações Rápidas do Agente")
        if st.button("🚀 Executar Treinamento Completo (Re-treinar Modelo)", use_container_width=True):
            with st.spinner("Extraindo cinemática e refinando modelo Random Forest..."):
                agent = TableTennisDatasetAgent()
                result = agent.train_model()
                st.success(f"Modelo re-treinado com sucesso! Nova Acurácia: {result['accuracy_percentage']}%")
                st.rerun()

        if st.button("📥 Minerar / Gerar Vídeos de Movimentos de Tênis de Mesa", use_container_width=True):
            with st.spinner("Minerando e gerando clipes biomecânicos de forehand, backhand, smash, chop e saque..."):
                agent = TableTennisDatasetAgent()
                videos = agent.fetch_videos(count_per_stroke=1)
                st.success(f"{len(videos)} vídeos adquiridos e indexados com sucesso!")

    with col_b:
        st.subheader("📁 Amostras do Dataset Processado")
        csv_path = os.path.join("data", "processed", "dataset_strokes.csv")
        if os.path.exists(csv_path):
            df_preview = pd.read_csv(csv_path)
            st.caption(f"Total de frames analisados: **{len(df_preview)}** amostras estruturadas.")
            st.dataframe(df_preview.head(25), height=250, use_container_width=True)
        else:
            st.info("Dataset ainda não gerado. Clique no botão de treinamento para iniciar.")


# 7. View: Como Jogar
elif nav_mode == "📖 Como Jogar":
    render_header()
    st.subheader("📖 Guia Interativo de Usabilidade e Regras Oficiais")
    
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.markdown(
            """
            <div style="background: rgba(14, 25, 45, 0.85); border: 1.5px solid rgba(0, 229, 255, 0.35);
                        border-radius: 12px; padding: 18px; min-height: 230px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);">
                <h4 style="color: #00e5ff; margin-top: 0;">🖱️ Modo Mouse (60 FPS)</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    Ideal para jogar imediatamente sem qualquer calibração de câmera.
                </p>
                <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 20px;">
                    <li><b>Movimentação:</b> Mova o cursor do mouse livremente sobre a mesa.</li>
                    <li><b>Rebatida Suave:</b> A assistência arcade intercepta e devolve no tempo do quique.</li>
                    <li><b>Power Smash:</b> Clique com o botão esquerdo para golpear em alta velocidade.</li>
                    <li><b>Saque:</b> Clique com o mouse para iniciar o ponto quando for sua vez.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_g2:
        st.markdown(
            """
            <div style="background: rgba(14, 25, 45, 0.85); border: 1.5px solid rgba(0, 230, 118, 0.35);
                        border-radius: 12px; padding: 18px; min-height: 230px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);">
                <h4 style="color: #00e676; margin-top: 0;">✋ Modo Webcam (MediaPipe)</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    Rastreamento contínuo dos 21 pontos da mão e articulação do braço.
                </p>
                <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 20px;">
                    <li><b>Empunhadura Real:</b> Feche a mão como se segurasse a raquete (Shakehand, Caneta ou Classineta).</li>
                    <li><b>Posicionamento:</b> Fique a ~1.5m da webcam com iluminação frontal.</li>
                    <li><b>Golpe Físico:</b> Acelere o antebraço para frente para aplicar topspin.</li>
                    <li><b>Cortada (Smash):</b> Eleve a mão acima do ombro e golpeie de cima para baixo.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_g3:
        st.markdown(
            """
            <div style="background: rgba(14, 25, 45, 0.85); border: 1.5px solid rgba(255, 215, 0, 0.35);
                        border-radius: 12px; padding: 18px; min-height: 230px; box-shadow: 0 8px 24px rgba(0,0,0,0.3);">
                <h4 style="color: #ffd700; margin-top: 0;">⌨️ Teclas de Atalho & Menus</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem;">
                    Acessibilidade total por teclado nos menus e na partida.
                </p>
                <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 20px;">
                    <li><b>[M]:</b> Abrir / Fechar Menu de Lobby e Pausa.</li>
                    <li><b>[TAB] / Setas:</b> Navegar entre botões e abas com foco visual.</li>
                    <li><b>[ESPAÇO] / [ENTER]:</b> Confirmar botão selecionado / Sacar a bola.</li>
                    <li><b>[1] a [4]:</b> Trocar dificuldade (Fácil, Médio, Difícil, Campeão).</li>
                    <li><b>[R]:</b> Reiniciar placar da partida.</li>
                    <li><b>[F]:</b> Alternar modo Tela Cheia.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.divider()
    with st.expander("📜 Regras Oficiais de Tênis de Mesa (ITTF) Adotadas no Jogo", expanded=False):
        st.markdown(
            """
            - **Sistema de Sets:** Vence quem fizer 11 pontos primeiro com pelo menos 2 pontos de vantagem.
            - **Alternância de Saque:** O saque alterna a cada 2 pontos (ou a cada ponto em caso de empate 10x10).
            - **Quiques Válidos:** A bola deve quicar uma vez em cada lado da mesa durante o saque e uma vez no lado adversário durante o rally.
            - **Let (Rede no Saque):** Se a bola tocar a fita da rede no saque e cair na mesa adversária, o saque é repetido sem penalidade.
            """
        )
