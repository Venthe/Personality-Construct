from datetime import datetime
import time
from flask import Flask, request, send_file, g
from text_to_speech.utilities.converter import audio_buffer_to_mp3
from text_to_speech.tone_converter.tone_converter_trainer import ToneConverter
from text_to_speech.text_to_speech.text_to_speech import TextToSpeech
from .config_wrapped_api import (
    prepare_embedding,
    embedder as _embedder,
    text_to_speech as _text_to_speech,
    text_to_speech_generate,
)
import soundfile
from . import config

app = Flask(__name__)
text_to_speech: TextToSpeech = _text_to_speech()
embedder: ToneConverter = _embedder()
server_config = config.TextToSpeechConfiguration().server


def convert(text):
    print(f"TTS: {text}")

    start_time = time.time()

    print(text_to_speech, text_to_speech.__dict__)
    sound_file_buffer, sampling_rate = text_to_speech_generate(text_to_speech, text)
    embedding = prepare_embedding(embedder=embedder)
    sound_file_buffer, sampling_rate = embedding(sound_file_buffer, sampling_rate)

    print(f"Text converted to sound in {time.time() - start_time}s")
    return audio_buffer_to_mp3(sound_file_buffer)


# Invoke-RestMethod -Uri http://localhost:5000/text-to-speech -Method Post -ContentType "text/plain" -Body "Your text here"
@app.route("/text-to-speech", methods=["POST"])
def _text_to_speech():
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
