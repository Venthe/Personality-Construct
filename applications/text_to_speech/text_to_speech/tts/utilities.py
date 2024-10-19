import soundfile
import io
import librosa


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
