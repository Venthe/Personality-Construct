import logging
import queue
import time
import numpy
import sounddevice


class MicrophoneListener:
    def __init__(
        self,
        sample_rate=16000,
        block_size=1500,
        silence_threshold=0.007,
        input_device_index=None,
        split_silence_duration_seconds=0.8,
    ):
        self._logger = logging.getLogger(__name__)
        self._audio_queue = queue.Queue()

        self._sample_rate = sample_rate
        self._block_size = block_size
        self._silence_threshold = silence_threshold
        self._input_device_index = self._select_input_device(input_device_index)
        self._split_silence_duration_seconds = split_silence_duration_seconds

    def _select_input_device(self, default_input_device_index=None):
        # List available audio input devices.
        self._logger.debug("Available audio input devices:")
        self._logger.debug(sounddevice.query_devices())
        input_device_index = (
            default_input_device_index
            if default_input_device_index is not None
            else sounddevice.default.device[0]
        )
        self._logger.debug(f"Selected microphone index: {input_device_index}")
        return input_device_index

    def listen(self, sound_recognized_callback=None):
        total_buffer = numpy.empty((0, 1), dtype=numpy.float32)
        current_buffer = numpy.empty((0, 1), dtype=numpy.float32)
        last_silence_time = time.time()
        speech_detected = False

        # TODO: Write a cleaner "clean" functions without so many variables
        with sounddevice.InputStream(
            samplerate=self._sample_rate,
            device=self._input_device_index,
            channels=1,
            blocksize=self._block_size,
            callback=self._audio_callback,
        ):
            self._logger.info(
                f"Listening to microphone {self._get_sound_device_name(self._input_device_index)}"
            )
            while True:
                audio_data = self._audio_queue.get()
                # We use the current buffer to not append unnecessary silence to the samples
                current_buffer = numpy.append(current_buffer, audio_data)

                if not self._detect_silence_and_transcribe(audio_data):
                    self._logger.debug("Speech detected")
                    speech_detected = True
                    last_silence_time = time.time()
                    total_buffer = numpy.append(total_buffer, current_buffer)
                    current_buffer = numpy.empty((0, 1), dtype=numpy.float32)

                silence_duration = time.time() - last_silence_time
                if silence_duration <= self._split_silence_duration_seconds:
                    self._logger.debug(
                        f"Silence detected for duration {silence_duration}"
                    )
                    continue

                if not speech_detected:
                    # If the buffer is empty, reset the buffer
                    self._logger.debug("No speech detected in the buffer, resetting")
                    total_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                    current_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                    last_silence_time = time.time()
                    speech_detected = False
                    continue

                # Process the audio buffer, capturing additional preceding audio
                self._logger.info(
                    f"Queueing fragment of length {self._get_buffer_length(total_buffer)}s"
                )
                self._logger.debug(f"Queueing fragment of {len(total_buffer)} samples")

                # TODO: Split very long audio files into smaller chunks
                if sound_recognized_callback:
                    sound_recognized_callback(total_buffer)

                total_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                current_buffer = numpy.empty((0, 1), dtype=numpy.float32)
                last_silence_time = time.time()
                speech_detected = False

    def _audio_callback(self, indata, frames, time, status):
        if status:
            self._logger.info(f"Audio stream status: {status}")
        self._audio_queue.put(indata.copy())

    def _get_sound_device_name(self, input_device_index):
        return sounddevice.query_devices()[input_device_index]["name"]

    def _detect_silence_and_transcribe(self, buffer):
        energy = numpy.mean(numpy.abs(buffer))
        self._logger.debug(f"Energy: {energy}")
        return energy < self._silence_threshold

    def _get_buffer_length(self, buffer):
        length_in_seconds = len(buffer) / self._sample_rate
        return length_in_seconds
