"""Распознавание речи: запись с микрофона → текст (Whisper через faster-whisper).

Зависимости ставятся отдельно: pip install -e ".[voice]" (faster-whisper, sounddevice).
Модель `small` (244 M параметров) распознаёт русский и английский на CPU за ~1–2 с на
фразу; веса скачиваются при первом вызове (~460 МБ).

Whisper — энкодер-декодер трансформер над лог-мел-спектрограммой (L7 §7.5); здесь он
используется как чёрный ящик: нас интересует текст, который дальше разбирает
`rwm.language.commands.parse`.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000


def record(seconds: float = 4.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Записать моно-сигнал с микрофона по умолчанию; float32 в [−1, 1]."""
    import sounddevice as sd

    print(f"запись {seconds:.0f} с ...", flush=True)
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "cpu", language: str | None = None):
        from faster_whisper import WhisperModel

        compute = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_size, device=device, compute_type=compute)
        self.language = language

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language=self.language, beam_size=5)
        return " ".join(s.text.strip() for s in segments)


def listen(seconds: float = 4.0, transcriber: Transcriber | None = None) -> str:
    transcriber = transcriber or Transcriber()
    return transcriber(record(seconds))
