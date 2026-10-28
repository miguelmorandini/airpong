"""Streamlit UI components, KPI metrics cards and Plotly analytics charts."""

from typing import List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.engine.game_state import MatchStats, ShotRecord


def render_header():
    """Render top page header and badges."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between;
                    background: linear-gradient(90deg, #0e1e38 0%, #1a3a60 100%);
                    padding: 16px 24px; border-radius: 12px; margin-bottom: 20px;
                    border: 1px solid #2a5280; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <div>
                <h1 style="color: #ffffff; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">
                    🏓 AI Ping Pong
                </h1>
                <p style="color: #88c0d0; margin: 4px 0 0 0; font-size: 0.95rem;">
                    Wii Sports Table Tennis &bull; Hand Tracking &bull; Databricks Apps
                </p>
            </div>
            <div style="text-align: right;">
                <span style="background-color: #ff3621; color: white; padding: 6px 12px;
                             border-radius: 20px; font-size: 0.8rem; font-weight: 700; margin-right: 8px;">
                    DATABRICKS APPS
                </span>
                <span style="background-color: #00b4d8; color: white; padding: 6px 12px;
                             border-radius: 20px; font-size: 0.8rem; font-weight: 700;">
                    MEDIAPIPE 60 FPS
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(stats: MatchStats, player_score: int, ai_score: int, server: str):
    """Render real-time match KPI cards."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Placar da Partida",
            value=f"{player_score} - {ai_score}",
            delta="Você vs Roberval",
        )

    with c2:
        st.metric(
            label="Serviço Atual",
            value="Jogador (Você)" if server == "PLAYER" else f"Roberval ({stats.difficulty})",
            delta="Alterna a cada 2 pontos",
        )

    with c3:
        st.metric(
            label="Maior Rally",
            value=f"{stats.longest_rally} golpes",
            delta=f"Total: {stats.total_rallies} trocas",
        )

    with c4:
        st.metric(
            label="Velocidade Média",
            value=f"{stats.avg_speed_kmh:.1f} km/h",
            delta=f"Smashes: {stats.player_smashes}",
        )


def render_shot_heatmap(shots: List[ShotRecord]):
    """Render 2D interactive table landing heatmap using Plotly."""
    if not shots:
        st.info("Nenhuma rebatida registrada ainda. Inicie o jogo para visualizar o heatmap.")
        return

    # Filter shots that bounced on table
    landing_shots = [s for s in shots if s.landing_x is not None and s.landing_z is not None]
    if not landing_shots:
        st.info("Aguardando os primeiros quiques na mesa para gerar o mapa de calor...")
        return

    data = []
    for s in landing_shots:
        data.append({
            "X (m)": s.landing_x,
            "Z (m)": s.landing_z,
            "Golpe": s.shot_type,
            "Velocidade (km/h)": round(s.speed_kmh, 1),
            "Jogador": "Humano" if s.hitter == "PLAYER" else "IA",
            "Resultado": s.result,
        })

    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="X (m)",
        y="Z (m)",
        color="Jogador",
        size="Velocidade (km/h)",
        hover_data=["Golpe", "Velocidade (km/h)", "Resultado"],
        color_discrete_map={"Humano": "#00b4d8", "IA": "#ff3621"},
        title="<b>Mapa de Impacto da Bola na Mesa (2D Heatmap)</b>",
    )

    # Table bounds lines (ITTF: X in [-0.7625, 0.7625], Z in [-1.37, 1.37])
    fig.add_shape(
        type="rect",
        x0=-0.7625, y0=-1.37, x1=0.7625, y1=1.37,
        line=dict(color="#ffffff", width=2),
        fillcolor="rgba(16, 92, 168, 0.25)",
    )

    # Net line at Z = 0
    fig.add_shape(
        type="line",
        x0=-0.90, y0=0.0, x1=0.90, y1=0.0,
        line=dict(color="#ffff00", width=3, dash="dash"),
    )

    # Center dividing line
    fig.add_shape(
        type="line",
        x0=0.0, y0=-1.37, x1=0.0, y1=1.37,
        line=dict(color="#ffffff", width=1, dash="dot"),
    )

    fig.update_layout(
        xaxis=dict(range=[-1.0, 1.0], title="Largura da Mesa (m)"),
        yaxis=dict(range=[-1.6, 1.6], title="Comprimento da Mesa (m)"),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_shot_type_distribution(shots: List[ShotRecord]):
    """Render shot distribution pie chart."""
    if not shots:
        return

    data = [{"Tipo": s.shot_type, "Jogador": "Humano" if s.hitter == "PLAYER" else "IA"} for s in shots]
    df = pd.DataFrame(data)

    fig = px.pie(
        df,
        names="Tipo",
        title="<b>Distribuição de Tipos de Golpe</b>",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def render_speed_history(shots: List[ShotRecord]):
    """Render speed timeline across rally."""
    if not shots:
        return

    data = [{
        "Golpe #": i + 1,
        "Velocidade (km/h)": s.speed_kmh,
        "Jogador": "Humano" if s.hitter == "PLAYER" else "IA",
    } for i, s in enumerate(shots)]

    df = pd.DataFrame(data)
    fig = px.line(
        df,
        x="Golpe #",
        y="Velocidade (km/h)",
        color="Jogador",
        markers=True,
        title="<b>Evolução da Velocidade das Rebatidas</b>",
        color_discrete_map={"Humano": "#00b4d8", "IA": "#ff3621"},
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
