import io
import ffmpeg


def audio_buffer_to_mp3(audio_buffer):
    """Convert a WAV file to MP3 and output to a memory buffer."""
    audio_buffer.seek(0)
    mem_file = io.BytesIO()
    process = (
        ffmpeg.input("pipe:", format="wav")
        .output("pipe:", format="mp3")  # Output to pipe with MP3 format
        .run(input=audio_buffer.read(), capture_stdout=True, capture_stderr=True)
    )
    mem_file.write(process[0])
    mem_file.seek(0)

    return mem_file
