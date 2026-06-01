"""Microphone recording and audio playback via sounddevice / soundfile."""

import sounddevice as sd
import soundfile as sf
import numpy as np


class AudioHandler:
    """Manages system audio input (record) and output (playback).

    Recording is streamed into an in-memory buffer and flushed to a WAV
    file when stopped.  Playback is synchronous (blocks until done).
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_chunks: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None

    def start_recording(self, callback_status=None):
        """Open a microphone stream and begin buffering audio."""
        self.recording = True
        self.audio_chunks = []

        def _cb(indata, frames, time, status):
            if self.recording:
                self.audio_chunks.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=_cb,
        )
        self.stream.start()

    def stop_recording(self, output_path):
        """Stop the stream and write the buffered audio to *output_path*.

        Returns True if audio was captured, False otherwise.
        """
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_chunks:
            audio = np.concatenate(self.audio_chunks, axis=0)
            sf.write(output_path, audio, self.sample_rate)
            return True
        return False

    def play_audio(self, file_path):
        """Play a WAV file (blocks until playback finishes)."""
        data, fs = sf.read(file_path)
        sd.play(data, fs)
        sd.wait()
