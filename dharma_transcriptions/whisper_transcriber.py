import os
import whisper
import torch

# Caminho do modelo treinado
TRAINED_MODEL_PATH = os.path.join("trained_models", "whisper_finetuned.pt")

def load_finetuned_model():
    """
    Carrega o modelo Whisper treinado (fine-tuned).
    """
    if not os.path.exists(TRAINED_MODEL_PATH):
        raise FileNotFoundError(f"Modelo treinado não encontrado: {TRAINED_MODEL_PATH}")
    
    print("[INFO] Carregando modelo treinado...")
    model = whisper.load_model("base")
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    print("[INFO] Modelo treinado carregado com sucesso.")
    return model


def transcribe_with_finetuned_model(audio_path, output_dir):
    """
    Transcreve o áudio usando o modelo treinado e salva os resultados.

    Args:
        audio_path (str): Caminho para o arquivo de áudio.
        output_dir (str): Diretório de saída para os resultados.

    Returns:
        dict: Caminhos para a transcrição e o arquivo de legendas.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
    
    # Certifique-se de que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)

    # Carregar o modelo treinado
    model = load_finetuned_model()

    print(f"[INFO] Transcrevendo o áudio: {audio_path}...")
    result = model.transcribe(audio_path, fp16=False)
    print("[INFO] Transcrição concluída com sucesso.")

    # Salvar transcrição
    transcription_path = os.path.join(output_dir, "transcription.txt")
    with open(transcription_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"[INFO] Transcrição salva em: {transcription_path}")

    # Salvar legendas
    subtitle_path = os.path.join(output_dir, "subtitles.srt")
    with open(subtitle_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            start = format_time(segment["start"])
            end = format_time(segment["end"])
            text = segment["text"]
            f.write(f"{i + 1}\n{start} --> {end}\n{text}\n\n")
    print(f"[INFO] Subtítulos salvos em: {subtitle_path}")

    return {"transcription_path": transcription_path, "subtitle_path": subtitle_path}


def format_time(seconds):
    """
    Converte tempo em segundos para o formato SRT (HH:MM:SS,mmm).

    Args:
        seconds (float): Tempo em segundos.

    Returns:
        str: Tempo formatado como HH:MM:SS,mmm.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


if __name__ == "__main__":
    # Exemplo de uso
    audio_path = "TREINAMENTO/audio/Prece das Sete Linhas.mp3"
    output_dir = "TREINAMENTO/output"

    try:
        results = transcribe_with_finetuned_model(audio_path, output_dir)
        print("[INFO] Resultados:")
        print(f" - Transcrição: {results['transcription_path']}")
        print(f" - Legendas: {results['subtitle_path']}")
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro: {e}")
