import os
import yt_dlp
from dharma_transcriptions.utils import sanitize_filename

def download_audio(youtube_url):
    output_folder = "downloads"
    os.makedirs(output_folder, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_folder, "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            },
        ],
        "ffmpeg_location": "C:/PATH_Programs/ffmpeg.exe"
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            raw_title = info_dict.get("title", "arquivo_desconhecido")
            sanitized_title = sanitize_filename(raw_title)
            audio_file = os.path.join(output_folder, f"{sanitized_title}.mp3")
            
            # Renomeie o arquivo após o download e sanitização
            downloaded_file = ydl.prepare_filename(info_dict).replace(".webm", ".mp3")
            os.rename(downloaded_file, audio_file)

            return audio_file  # Retorna apenas o caminho do arquivo
    except Exception as e:
        raise Exception(f"Erro ao baixar ou converter áudio: {str(e)}")

