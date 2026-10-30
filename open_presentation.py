"""Helper script to launch the AI Ping Pong Technical Slide Presentation."""
import webbrowser
from pathlib import Path

def main():
    presentation_file = Path(__file__).parent / "presentation" / "index.html"
    if presentation_file.exists():
        url = presentation_file.resolve().as_uri()
        print(f"[AI Ping Pong] Abrindo apresentacao tecnica: {url}")
        webbrowser.open(url)
    else:
        print(f"[ERRO] Arquivo de apresentacao nao encontrado em: {presentation_file}")

if __name__ == "__main__":
    main()
