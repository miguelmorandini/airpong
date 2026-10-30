# 🏓 AI Ping Pong

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-success?logo=github)](https://miguelmorandini.github.io/airpong/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://python.org)
[![Databricks Apps](https://img.shields.io/badge/Databricks-Apps%20%26%20Delta%20Lake-FF3621?logo=databricks)](https://www.databricks.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-60%20FPS%20Hand%20Tracking-00A389?logo=google)](https://developers.google.com/mediapipe)
[![Tests](https://img.shields.io/badge/Tests-49%20Passed%20%28100%25%29-brightgreen)](#testes-automatizados)

**AI Ping Pong** é um jogo de tênis de mesa em perspectiva 3D inspirado no clássico **Wii Sports**, controlado em tempo real pelos movimentos da mão capturados pela webcam, com adversário controlado por Inteligência Artificial, motor de física avançado (aerodinâmica, quiques ITTF e efeito Magnus/spin), interface em **Streamlit** e arquitetura pronta para implantação no **Databricks Apps** com persistência analítica em tabelas **Delta Lake**.

---

## 🌟 Principais Funcionalidades

1. **Controle por Visão Computacional (MediaPipe Hands & OpenCV)**:
   - Rastreamento dos 21 landmarks da mão a 60 FPS.
   - Estimativa de profundidade ($Z$) baseada na escala métrica da mão no campo visual.
   - Cálculo vetorial da velocidade do swing ($\Delta \vec{p} / \Delta t$) com suavização exponencial (EMA) para eliminação de ruído.
   - Classificação automática de golpes: **Forehand**, **Backhand**, **Power Smash** e **Drive**.

2. **Motor de Física 3D & Regras Oficiais ITTF**:
   - Geometria oficial da mesa ITTF (2,74 m de comprimento $\times$ 1,525 m de largura $\times$ 0,1525 m de altura de rede).
   - Resistência do ar aerodinâmica (arrasto quadrático) e gravidade contínua.
   - **Efeito Magnus (Spin)**: topspin que faz a bola mergulhar, backspin para sustentação e sidespin para desvios laterais.
   - Regras oficiais: sets de 11 pontos, vantagem mínima de 2 pontos (deuce), alternância de saque a cada 2 pontos (ou 1 ponto no deuce) e detecção de faltas de rede e mesa.

3. **Adversário Inteligente (IA)**:
   - 4 níveis de dificuldade configuráveis: **EASY**, **MEDIUM**, **HARD** e **CHAMPION**.
   - Predição balística de ponto de interceptação na linha de rebatida.
   - Variação estocástica de erro e tempo de reação humano simulado.

4. **Estética Inspirada no Wii Sports**:
   - Projeção de perspectiva 3D realista com iluminação e sombras projetadas.
   - Rastro de velocidade da bola (*speed trail*), faíscas de impacto e feedbacks visuais em pop-up ("NICE SHOT!", "SMASH!", "OUT!", "DEUCE!").
   - Placar estilizado com indicador visual do sacador.

5. **Hub Analítico e Telemetria no Databricks**:
   - Persistência estruturada em tabelas Delta Lake (`ping_pong_matches` e `ping_pong_shots`).
   - Mapa de calor 2D (*Heatmap*) dos pontos de impacto da bola na mesa.
   - Gráficos interativos em Plotly para distribuição de tipos de golpe e curvas de aceleração.
   - Fallback resiliente para SQLite local: funciona 100% mesmo sem credenciais ativas do Databricks.

---

## 🏗️ Arquitetura do Sistema

O projeto adota uma **Arquitetura Híbrida / Multi-Target**, projetada especificamente para contornar as limitações físicas de captura de webcam e latência em contêineres de nuvem:

```mermaid
flowchart TD
    subgraph LocalClient["1. Camada de Execução Local (Zero Latência)"]
        Webcam["Webcam Física (USB/Integrada)"] --> MPTracker["MediaPipe Hands (60 FPS)"]
        MPTracker --> Kinematics["Cinemática & Reconhecimento de Golpes"]
        Kinematics --> Engine["Motor de Física 3D (Colisões, Arrasto, Spin)"]
        Engine --> Renderer3D["Renderizador de Perspectiva 3D (Wii Sports HUD)"]
    end

    subgraph DatabricksAppsCloud["2. Nuvem Databricks / Databricks Apps"]
        StreamlitCloud["Databricks App (Streamlit Server)"]
        WebPlayground["Playground Jogável (Virtual / Sliders / Simulação)"]
        TelemetryPipeline["Ingestão de Telemetria (SDK / REST / SQL)"]
        DeltaTables[("Tabelas Delta Lake:\nping_pong_matches\nping_pong_shots")]
        AnalyticsDash["Dashboard de Estatísticas & Heatmaps"]
    end

    Engine -- "JSON / SQLite Fallback" --> TelemetryPipeline
    TelemetryPipeline --> DeltaTables
    DeltaTables --> AnalyticsDash
    StreamlitCloud --> AnalyticsDash
    StreamlitCloud --> WebPlayground
```

### Por que esta arquitetura?
Em um jogo com trocas de bola que ocorrem entre 400 ms e 700 ms, o streaming de vídeo do browser para uma nuvem remota gera entre 250 ms e 450 ms de latência total de ida e volta, tornando o jogo lento. Nossa arquitetura oferece:
- **Modo Desktop / Local**: Latência inferior a 15 ms com renderização direta e gravação de telemetria.
- **Modo Databricks Apps**: Hub central de torneios, heatmaps, estatísticas avançadas e modo jogável interativo em nuvem homologado para qualquer navegador corporativo.

---

## 📁 Estrutura do Projeto

```
ping-pong/
├── .github/
│   └── workflows/
│       ├── ci.yml               # Pipeline de testes e integração contínua (CI)
│       └── pages.yml            # Pipeline de deploy automático no GitHub Pages
├── .gitignore                   # Arquivos ignorados pelo Git
├── README.md                    # Documentação do projeto
├── docs/                        # Build Web 100% Client-Side para GitHub Pages
│   ├── index.html               # Canvas 3D e carregamento MediaPipe Hands CDN
│   ├── style.css                # Estilização Cyber-Stadium e interface responsiva
│   └── game.js                  # Física 3D, IA Roberval, áudio procedural e visão Web
├── app.yaml                     # Manifesto de implantação no Databricks Apps
├── requirements.txt             # Dependências de produção
├── requirements-dev.txt         # Dependências de teste e desenvolvimento
├── pyproject.toml               # Configuração do pacote Python e pytest
├── app.py                       # Aplicação Streamlit principal para Databricks Apps
├── main.py                      # Runner Desktop autônomo nativo em 60 FPS (OpenCV)
├── src/
│   ├── __init__.py
│   ├── config.py                # Dimensões ITTF, constantes físicas e parâmetros
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── table.py             # Geometria e limites da mesa 3D
│   │   ├── paddle.py            # Cinemática e colisão da raquete virtual
│   │   ├── physics.py           # Arrasto aerodinâmico, efeito Magnus e quiques
│   │   ├── ai.py                # Oponente IA com 4 níveis de dificuldade
│   │   └── game_state.py        # Máquina de estados da partida e pontuação
│   ├── dataset_agent/
│   │   ├── __init__.py
│   │   ├── video_miner.py       # Mineração de vídeos no YouTube e gerador sintético biomecânico
│   │   ├── hand_extractor.py    # Extração de cinemática de 21 landmarks via MediaPipe
│   │   ├── dataset_builder.py   # Agregação, aumento e exportação (CSV, Parquet, JSON)
│   │   ├── model_trainer.py     # Treinador Random Forest & avaliação de acurácia
│   │   └── agent.py             # Agente autônomo CLI e orquestrador ponta a ponta
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── tracker.py           # Rastreamento de mãos via MediaPipe Hands & Pose
│   │   ├── camera.py            # Abstrações de câmera (Webcam, Câmera Sintética)
│   │   ├── calibration.py       # Calibração e perfil personalizado de movimentos
│   │   ├── gestures.py          # Classificador híbrido ML + Heurística
│   │   └── one_euro_filter.py   # Filtro 1€ para eliminação de jitter em tempo real
│   ├── databricks/
│   │   ├── __init__.py
│   │   ├── client.py            # Conector de autenticação do Databricks
│   │   ├── storage.py           # Camada unificada para Delta Lake no Unity Catalog
│   │   ├── mock_storage.py      # Armazenamento SQLite com schema idêntico ao Delta
│   │   └── analytics.py         # Cálculos de erros não-forçados e velocidades
│   └── ui/
│       ├── __init__.py
│       ├── renderer_3d.py       # Renderizador de perspectiva 3D estilo Wii Sports
│       ├── sound_effects.py     # Gerenciador de impactos visuais e faíscas
│       └── components.py        # Componentes do Streamlit e gráficos Plotly
└── tests/
    ├── __init__.py
    ├── test_physics.py          # Testes de limites de mesa e colisão
    ├── test_game_rules.py       # Testes de regras ITTF, saque e deuce
    ├── test_ai.py               # Testes de interceptação da IA
    ├── test_vision_tracker.py   # Testes do pipeline de visão e câmera sintética
    ├── test_storage.py          # Testes de persistência Delta/SQLite e analytics
    └── test_integration.py      # Teste de integração de ponta a ponta
```

---

## 🌐 Jogar Online no Navegador (GitHub Pages - Sem Instalação)

A versão Web do **AI Ping Pong** roda **100% no navegador (Client-Side)** usando **HTML5 Canvas 3D**, **Web Audio API** e **MediaPipe Hands via WebAssembly/CDN**. Não requer instalação de Python, servidores backend ou Docker!

- **Demo Online**: `https://miguelmorandini.github.io/airpong/`
- **Rastreamento de Mão**: Captura movimentos na webcam direto no navegador com privacidade total (o vídeo não sai do seu computador).
- **Controle Híbrido**: Alterne a qualquer momento entre a webcam e o mouse.
- **Áudio Procedural**: Efeitos sonoros gerados em tempo real via Web Audio API.

### Como Ativar o GitHub Pages no seu Repositório:
1. No seu repositório no GitHub, clique na aba **Settings** (Configurações).
2. No menu lateral esquerdo, clique em **Pages**.
3. Em **Build and deployment > Source**, selecione:
   - **Opção A (Recomendada / Instantânea)**: **Deploy from a branch**
     - Branch: `master` (ou `main`)
     - Pasta: `/docs`
     - Clique em **Save**.
   - **Opção B (GitHub Actions)**: **GitHub Actions**
     - O workflow [.github/workflows/pages.yml](file:///.github/workflows/pages.yml) fará o deploy automático a cada `git push`!
4. Em 1 a 2 minutos, o GitHub exibirá a URL pública do seu jogo pronta para compartilhar!

### Como Testar a Versão Web Localmente:
```bash
# Na raiz do projeto:
python -m http.server 8000 --directory docs
```
Acesse `http://localhost:8000` no Google Chrome, Edge ou Firefox.

---

## 🚀 Instalação e Execução Local

### 1. Pré-requisitos
- Python 3.10, 3.11 ou 3.12 instalado.
- Webcam conectada (para controle por visão computacional).

### 2. Instalar Dependências
```bash
# Clonar repositório
git clone https://github.com/miguelmorandini/airpong.git
cd airpong

# Criar e ativar ambiente virtual (opcional, recomendado)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. Execução no Desktop (Modo Nativo 60 FPS com Mouse ou Webcam)
Para a experiência arcade nativa com máxima fluidez:
```bash
python main.py
```
**Controles Intuitivos:**
- 🖱️ **Mouse / Cursor**: Mova o mouse sobre a janela para mover a raquete com resposta instantânea e precisa.
- 🖱️ **Clique Esquerdo**: Sacar a bola ou disparar um *Power Smash* no momento do impacto.
- ✋ **Webcam / Mão**: O sistema detecta os 21 pontos da sua mão automaticamente.
- 🏓 **Assistência Wii Sports**: A raquete possui zona de atração de 0,50 m para garantir rebatidas fáceis e ralis longos e divertidos.
- `[ESPAÇO]`: Sacar a bola
- `[R]`: Reiniciar partida
- `[1]`, `[2]`, `[3]`, `[4]`: Mudar dificuldade da IA (Easy, Medium, Hard, Champion)
- `[S]`: Salvar estatísticas da partida
- `[Q]` ou `[ESC]`: Sair

### 4. Execução Web no Streamlit (Databricks Apps)
Para executar a quadra 3D interativa de 60 FPS no navegador:
```bash
streamlit run app.py
```
Acesse no navegador: `http://localhost:8501`.
- A mesa 3D roda a **60 FPS diretamente no browser via HTML5 Canvas**.
- Basta mover o cursor do mouse sobre a quadra para rebater a bola com física em tempo real e efeitos sonoros!
- **Aba "Dataset & Treino de Movimentos"**: Visualize estatísticas do modelo ML, classes de golpes, minerador de vídeos e execute re-treinamento com 1 clique!

### 5. 🤖 Agente de Mineração de Vídeos e Treinamento de Movimentos (Dataset Agent)
O projeto conta com um **Agente Autônomo** para minerar vídeos de movimentos reais de tênis de mesa (YouTube ou gerador sintético biomecânico), extrair cinemática de 21 landmarks via MediaPipe e treinar classificadores ML:
```bash
# Executar pipeline ponta a ponta (mineração + extração + dataset + treino ML)
python -m src.dataset_agent.agent pipeline --count 1

# Minerar vídeos específicos no YouTube ou gerar amostras
python -m src.dataset_agent.agent fetch --stroke GANCHO_FOREHAND --count 1

# Processar vídeos locais e gerar dataset estruturado
python -m src.dataset_agent.agent build

# Treinar e avaliar o modelo Random Forest
python -m src.dataset_agent.agent train
```

---

## ☁️ Implantação no Databricks Apps

O projeto inclui o manifesto `app.yaml` configurado para execução nativa no **Databricks Apps**:

```yaml
command:
  - "streamlit"
  - "run"
  - "app.py"
  - "--server.port"
  - "8501"
  - "--server.address"
  - "0.0.0.0"
  - "--server.headless"
  - "true"
```

### Passos de Implantação via Databricks CLI:
```bash
# 1. Configurar credenciais do workspace
databricks configure

# 2. Criar a aplicação no Databricks Apps
databricks apps create ai-ping-pong --spec app.yaml

# 3. Fazer o deploy dos arquivos
databricks sync . /Workspace/Users/seu-usuario/ai-ping-pong
databricks apps deploy ai-ping-pong --source-code-path /Workspace/Users/seu-usuario/ai-ping-pong
```

### Schema das Tabelas Delta Lake (Unity Catalog):
- **`ping_pong.telemetry.ping_pong_matches`**: Registra metadados da partida, pontuação final, vencedor, maior rally e velocidade média.
- **`ping_pong.telemetry.ping_pong_shots`**: Registra coordenadas exatas $(X, Z)$ de cada quique, tipo de golpe, velocidade em km/h e desfecho (In play, Winner, Out, Net fault).

---

## 🐳 Implantação com Docker

Para rodar o container do AI Ping Pong em qualquer ambiente de nuvem ou servidor:

```bash
# Construir a imagem Docker
docker build -t ai-ping-pong:latest .

# Executar o container localmente
docker run -p 8501:8501 ai-ping-pong:latest
```
Acesse `http://localhost:8501`.

---

## 📽️ Apresentação Técnica & Pitch Deck

O projeto inclui um slide deck executivo/técnico moderno em HTML5 com métricas, arquitetura e comparativos com o Wii Sports:

```bash
# Abrir apresentação técnica no navegador
python open_presentation.py
```
Ou abra diretamente `presentation/index.html`.

---

## 🧪 Testes Automatizados

O projeto conta com **49 testes automatizados** cobrindo todas as camadas críticas (física, visão computacional, calibração personalizada, filtros contra tremor, regras ITTF, agente de dataset com machine learning, persistência e integração de ponta a ponta):

```bash
# Executar suite completa
pytest tests -v
```

Resultado:
```
======================== 49 passed, 2 warnings in 29.22s ========================
```

---

## 📜 Licença
Distribuído sob licença MIT. Consulte o arquivo [LICENSE](file:///LICENSE) para mais detalhes.
