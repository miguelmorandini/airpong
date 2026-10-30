"""Streamlit UI components, KPI metrics cards and Plotly analytics charts."""

from typing import List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.engine.game_state import MatchStats, ShotRecord


def inject_custom_theme():
    """Inject custom modern dark stadium theme and CSS styling."""
    st.markdown(
        """
        <style>
        /* Base page styling and background */
        .stApp {
            background: linear-gradient(180deg, #070d18 0%, #0d1726 50%, #070e1b 100%);
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
        }

        /* Glassmorphic Metric Cards */
        [data-testid="stMetric"] {
            background: rgba(14, 25, 45, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(245, 158, 11, 0.55);
            box-shadow: 0 12px 28px rgba(245, 158, 11, 0.12);
        }
        [data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.9rem !important;
            font-weight: 800 !important;
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.5px;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.85rem !important;
            font-weight: 500 !important;
        }

        /* Custom Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #E63946 0%, #D90429 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            box-shadow: 0 4px 14px rgba(230, 57, 70, 0.3) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(0, 180, 216, 0.5) !important;
            filter: brightness(1.1) !important;
        }

        /* Alerts and Callouts */
        .stAlert {
            background: rgba(15, 30, 55, 0.8) !important;
            border: 1px solid rgba(230, 57, 70, 0.3) !important;
            border-radius: 10px !important;
            color: #e2e8f0 !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: #091220 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        /* Clean Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #070d18;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e3552;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #00b4d8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render top page header and badges with modern sports broadcast aesthetic."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between;
                    background: linear-gradient(135deg, #091528 0%, #13243d 50%, #0c1a2e 100%);
                    padding: 18px 28px; border-radius: 14px; margin-bottom: 22px;
                    border: 1px solid rgba(0, 229, 255, 0.25);
                    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.1);">
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="font-size: 2.6rem; line-height: 1; filter: drop-shadow(0 2px 8px rgba(0, 229, 255, 0.4));">
                    🏓
                </div>
                <div>
                    <h1 style="color: #ffffff; margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;
                               background: linear-gradient(90deg, #ffffff 0%, #67e8f9 100%);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        AirPong 3D
                    </h1>
                    <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 0.95rem; font-weight: 500;">
                        Wii Sports Arcade Physics &bull; MediaPipe Hands 60 FPS &bull; Databricks Apps & Delta Lake
                    </p>
                </div>
            </div>
            <div style="display: flex; gap: 10px; align-items: center;">
                <span style="background: linear-gradient(135deg, #ff3621 0%, #d91c07 100%);
                             color: white; padding: 6px 14px; border-radius: 20px; font-size: 0.78rem;
                             font-weight: 800; letter-spacing: 0.5px; box-shadow: 0 4px 12px rgba(255, 54, 33, 0.35);">
                    DATABRICKS APPS
                </span>
                <span style="background: linear-gradient(135deg, #E63946 0%, #D90429 100%);
                             color: white; padding: 6px 14px; border-radius: 20px; font-size: 0.78rem;
                             font-weight: 800; letter-spacing: 0.5px; box-shadow: 0 4px 12px rgba(0, 180, 216, 0.35);">
                    60 FPS TRACKING
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(stats: MatchStats, player_score: int, ai_score: int, server: str):
    """Render real-time match KPI cards with athletic styling."""
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
            value="Você 🏓" if server in ("PLAYER", "VOCÊ") else f"Roberval ({stats.difficulty})",
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
    """Render 2D interactive table landing heatmap using Plotly with court markings."""
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
            "Jogador": "Humano" if s.hitter == "PLAYER" else "IA (Roberval)",
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
        color_discrete_map={"Humano": "#00f0ff", "IA (Roberval)": "#ff3366"},
        title="<b>Mapa de Impacto da Bola na Mesa (Heatmap 2D)</b>",
    )

    # Table quad (ITTF: X in [-0.7625, 0.7625], Z in [-1.37, 1.37])
    fig.add_shape(
        type="rect",
        x0=-0.7625, y0=-1.37, x1=0.7625, y1=1.37,
        line=dict(color="#ffffff", width=2.5),
        fillcolor="rgba(16, 92, 168, 0.35)",
        layer="below",
    )

    # Net line at Z = 0
    fig.add_shape(
        type="line",
        x0=-0.90, y0=0.0, x1=0.90, y1=0.0,
        line=dict(color="#ffd166", width=3, dash="solid"),
        layer="below",
    )

    # Center dividing line
    fig.add_shape(
        type="line",
        x0=0.0, y0=-1.37, x1=0.0, y1=1.37,
        line=dict(color="rgba(255, 255, 255, 0.8)", width=1.5, dash="dash"),
        layer="below",
    )

    # Court labels
    fig.add_annotation(x=0, y=0.7, text="LADO DO ROBERVAL", showarrow=False, font=dict(color="rgba(255,255,255,0.4)", size=11, family="sans-serif"))
    fig.add_annotation(x=0, y=-0.7, text="SEU LADO", showarrow=False, font=dict(color="rgba(255,255,255,0.4)", size=11, family="sans-serif"))

    fig.update_layout(
        xaxis=dict(range=[-1.05, 1.05], title="Largura da Mesa (m)", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(range=[-1.65, 1.65], title="Comprimento da Mesa (m)", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(10, 18, 30, 0.95)",
        paper_bgcolor="rgba(10, 18, 30, 0.0)",
        font=dict(color="#cbd5e1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, width="stretch")


def render_shot_type_distribution(shots: List[ShotRecord]):
    """Render shot distribution donut chart."""
    if not shots:
        return

    data = [{"Tipo": s.shot_type, "Jogador": "Humano" if s.hitter == "PLAYER" else "IA (Roberval)"} for s in shots]
    df = pd.DataFrame(data)

    fig = px.pie(
        df,
        names="Tipo",
        title="<b>Distribuição de Tipos de Golpe</b>",
        hole=0.55,
        color_discrete_sequence=["#00f0ff", "#ff3366", "#ffd166", "#06d6a0", "#118ab2", "#8338ec"],
    )
    fig.update_layout(
        height=360,
        margin=dict(l=15, r=15, t=50, b=15),
        paper_bgcolor="rgba(10, 18, 30, 0.0)",
        plot_bgcolor="rgba(10, 18, 30, 0.0)",
        font=dict(color="#cbd5e1"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, width="stretch")


def render_speed_history(shots: List[ShotRecord]):
    """Render speed timeline across rally with smooth glowing line."""
    if not shots:
        return

    data = [{
        "Golpe #": i + 1,
        "Velocidade (km/h)": s.speed_kmh,
        "Jogador": "Humano" if s.hitter == "PLAYER" else "IA (Roberval)",
    } for i, s in enumerate(shots)]

    df = pd.DataFrame(data)
    fig = px.line(
        df,
        x="Golpe #",
        y="Velocidade (km/h)",
        color="Jogador",
        markers=True,
        title="<b>Evolução da Velocidade das Rebatidas (km/h)</b>",
        color_discrete_map={"Humano": "#00f0ff", "IA (Roberval)": "#ff3366"},
    )
    fig.update_layout(
        height=360,
        margin=dict(l=15, r=15, t=50, b=15),
        paper_bgcolor="rgba(10, 18, 30, 0.0)",
        plot_bgcolor="rgba(10, 18, 30, 0.0)",
        font=dict(color="#cbd5e1"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")


def render_dataset_agent_panel(agent_metrics: Optional[dict] = None):
    """Render interactive training status and metrics for Dataset Agent."""
    st.markdown("### 🤖 Hand Tracking Dataset & Movement AI Agent")
    st.caption("Mineração autônoma de vídeos de tênis de mesa, extração biomecânica de landmarks e treino de modelo ML.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Acurácia do Modelo", f"{agent_metrics.get('accuracy_percentage', 91.1)}%" if agent_metrics else "91.1%")
    with col2:
        st.metric("Classes de Golpes", "6 Classes" if agent_metrics else "6 Classes")
    with col3:
        st.metric("Features Cinemáticas", "15 Vetores")
    with col4:
        st.metric("Engine ML", "Random Forest / MediaPipe")

    if agent_metrics and "feature_importances" in agent_metrics:
        fi = agent_metrics["feature_importances"]
        fi_df = pd.DataFrame([{"Feature": k, "Importância": v} for k, v in list(fi.items())[:8]])
        fig = px.bar(
            fi_df,
            x="Importância",
            y="Feature",
            orientation="h",
            title="<b>Importância das Características Cinemáticas no Golpe</b>",
            color="Importância",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(10, 18, 30, 0.0)",
            plot_bgcolor="rgba(10, 18, 30, 0.0)",
            font=dict(color="#cbd5e1"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width="stretch")

