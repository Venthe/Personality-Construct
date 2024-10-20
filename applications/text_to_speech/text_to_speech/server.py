from datetime import datetime
import time
from flask import Flask, request, send_file, g
from text_to_speech.converter import convert_wav_to_mp3_memory
from .tts.wrapped_api import (
    prepare_embedding,
    embedder as _embedder,
    text_to_speech as _text_to_speech,
    text_to_speech_generate,
)
import soundfile
from . import config

app = Flask(__name__)
text_to_speech = _text_to_speech()
embedder = _embedder()
server_config = config.TextToSpeechConfig().server


def convert(text_to_voice):
    print(f"TTS: {text_to_voice}")

    start_time = time.time()

    sound_file_buffer, sampling_rate = text_to_speech_generate(
        text_to_speech, text_to_voice
    )
    embedding = prepare_embedding(embedder=embedder)
    sound_file_buffer, sampling_rate = embedding(sound_file_buffer, sampling_rate)
    sound_file, _ = soundfile.read(sound_file_buffer)

    print(f"Text converted to sound in {time.time() - start_time}s")
    return convert_wav_to_mp3_memory(sound_file)


# Invoke-RestMethod -Uri http://localhost:5000/text-to-speech -Method Post -ContentType "text/plain" -Body "Your text here"
@app.route("/text-to-speech", methods=["POST"])
def text_to_speech():
    app.logger.info("Received request for text-to-speech")
    data = request.get_data(as_text=True)

    result = convert(data)

    return send_file(
        result, mimetype="audio/mpeg", as_attachment=True, download_name="output.mp3"
    )


def main():
    print(f"Starting server at {server_config.host()}:{server_config.port()}")
    app.run(debug=True, host=server_config.host(), port=server_config.port())


if __name__ == "__main__":
    main()
