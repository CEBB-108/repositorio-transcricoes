import os
import shutil
from dharma_transcriptions.utils import sanitize_filename
from dharma_transcriptions.youtube import download_audio
from dharma_transcriptions.whisper_core import load_model

# Caminho base absoluto
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "TREINAMENTO"))
AUDIO_PATH = os.path.join(BASE_PATH, "audio")
REFERENCE_PATH = os.path.join(BASE_PATH, "referencia_corrigida")
BRUTOS_PATH = os.path.join(BASE_PATH, "dados_brutos")
CORRIGIDOS_PATH = os.path.join(BASE_PATH, "textos_corrigidos")

# Instancia o transcritor Whisper
whisper_transcriber = load_model(True)

def validate_and_download_audio():
    """
    Valida os arquivos em 'referencia_corrigida', baixa os áudios correspondentes
    e move para a pasta 'audio'.
    """
    print("[INFO] Validando e baixando áudios...")
    if not os.path.exists(REFERENCE_PATH):
        print(f"[ERRO] Caminho não encontrado: {REFERENCE_PATH}")
        return []

    reference_files = os.listdir(REFERENCE_PATH)
    print(f"[DEBUG] Arquivos em referencia_corrigida: {reference_files}")
    pairs = []

    for txt_file in reference_files:
        if not txt_file.endswith(".txt"):
            print(f"[AVISO] Ignorando arquivo não .txt: {txt_file}")
            continue

        txt_path = os.path.join(REFERENCE_PATH, txt_file)

        # Ler o conteúdo do arquivo
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[ERRO] Falha ao ler o arquivo {txt_file}: {e}")
            continue

        if len(lines) < 1:
            print(f"[AVISO] Arquivo vazio ou sem link do YouTube: {txt_file}")
            continue

        # Extrair link do YouTube
        youtube_url = lines[0].strip()

        # Baixar o áudio
        print(f"[INFO] Baixando áudio para: {txt_file}")
        try:
            downloaded_audio_path = download_audio(youtube_url)
            audio_filename = os.path.basename(downloaded_audio_path)

            # Mover o áudio para AUDIO_PATH
            final_audio_path = os.path.join(AUDIO_PATH, audio_filename)
            os.makedirs(AUDIO_PATH, exist_ok=True)
            shutil.move(downloaded_audio_path, final_audio_path)
            print(f"[INFO] Áudio movido para: {final_audio_path}")
        except Exception as e:
            print(f"[ERRO] Falha ao baixar ou mover áudio para {txt_file}: {e}")
            continue

        pairs.append((txt_file, audio_filename))

    print(f"[DEBUG] Pares válidos encontrados: {pairs}")
    return pairs


def prepare_data_for_training(pairs):
    """
    Prepara os pares de dados para o treinamento:
    - Gera transcrições brutas, se necessário.
    - Move para a pasta correta.
    """
    for txt_file, audio_file in pairs:
        txt_path = os.path.join(REFERENCE_PATH, txt_file)
        audio_path = os.path.join(AUDIO_PATH, audio_file)

        print(f"[INFO] Usando texto corrigido: {txt_file} para o áudio: {audio_file}")

        # Gerar e mover transcrição bruta
        bruto_file_name = f"{os.path.splitext(audio_file)[0]}.txt"
        bruto_file_path = os.path.join(BRUTOS_PATH, bruto_file_name)

        if not os.path.exists(bruto_file_path):
            print(f"[INFO] Gerando transcrição bruta para: {audio_file}")
            try:
                os.makedirs(BRUTOS_PATH, exist_ok=True)
                transcription, segments = whisper_transcriber.transcribe_audio(audio_path)
                with open(bruto_file_path, "w", encoding="utf-8") as f:
                    f.write(transcription)
                print(f"[INFO] Transcrição bruta salva em: {bruto_file_path}")
            except Exception as e:
                print(f"[ERRO] Falha ao transcrever áudio para {audio_file}: {e}")
        else:
            print(f"[AVISO] Transcrição bruta já existente: {bruto_file_path}")

def start_training():
    """
    Inicia o processo de treinamento usando os dados organizados.
    """
    print("[INFO] Validando e baixando áudios...")
    pairs = validate_and_download_audio()

    if not pairs:
        print("[ERRO] Nenhum par válido encontrado para treinamento.")
        return

    print("[INFO] Preparando dados para treinamento...")
    prepare_data_for_training(pairs)

    # Aqui você pode adicionar a lógica para treinar o modelo
    print("[INFO] Dados prontos. Inicie o treinamento do modelo aqui!")


if __name__ == "__main__":
    start_training()
