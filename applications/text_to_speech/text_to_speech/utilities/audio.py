import soundfile
import io
import librosa
import ffmpeg


def to_soundfile(audio_data, sampling_rate=44100, format="WAV"):
    audio_buffer = io.BytesIO()
    soundfile.write(audio_buffer, audio_data, sampling_rate, format=format)
    audio_buffer.seek(0)

    return audio_buffer, sampling_rate


def resample_audio(audio_buffer, target_sr=44100):
    audio_buffer.seek(0)
    audio_data, original_sampling_rate = soundfile.read(audio_buffer)

    # Resample the audio data
    resampled_audio = librosa.resample(
        audio_data, orig_sr=original_sampling_rate, target_sr=target_sr
    )

    return to_soundfile(resampled_audio, target_sr)


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