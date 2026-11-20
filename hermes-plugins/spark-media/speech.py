"""Offline Faster Whisper and Kokoro command providers for Hermes."""

import os
import sys
from pathlib import Path

os.environ.update(HF_HOME=str(Path(sys.executable).absolute().parents[2] / "hf"),
                  HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", OMP_NUM_THREADS="4")

import av
import numpy as np

MAX_AUDIO_BYTES = 18 * 1024**2
MAX_SAMPLES = 180 * 16_000


def transcribe(source, destination):
    from faster_whisper import WhisperModel
    if not source.is_file() or source.is_symlink() or not 0 < source.stat().st_size <= MAX_AUDIO_BYTES:
        raise ValueError("Voice note exceeds the local input limit.")
    resample = av.AudioResampler(format="s16", layout="mono", rate=16000)
    pcm = []
    count = 0
    with av.open(str(source)) as audio:
        for frame in audio.decode(audio=0):
            for chunk in resample.resample(frame):
                samples = chunk.to_ndarray()[0]
                count += samples.size
                if count > MAX_SAMPLES:
                    raise ValueError("Send a voice note under three minutes.")
                pcm.append(samples)
        for chunk in resample.resample(None):
            pcm.append(chunk.to_ndarray()[0])
    if not pcm:
        raise ValueError("The attachment contains no audio.")
    waveform = np.concatenate(pcm).astype(np.float32) / 32768.0
    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
    segments, _ = model.transcribe(waveform, beam_size=3, vad_filter=True, language=None)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise ValueError("No intelligible speech was detected.")
    destination.write_text(text + "\n")


def synthesize(source, destination):
    import torch
    from kokoro import KPipeline
    text = source.read_text(encoding="utf-8").strip()
    if not 0 < len(text) <= 2000:
        raise ValueError("Provide no more than 2000 characters for a spoken reply.")
    torch.set_num_threads(4)
    model = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
    parts = [part.audio.numpy() for part in model(text, voice="af_heart") if part.audio is not None]
    if not parts:
        raise ValueError("Kokoro could not pronounce this text.")
    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype(np.int16)
    codec = {".wav": "pcm_s16le", ".ogg": "libopus", ".opus": "libopus", ".mp3": "libmp3lame"}.get(destination.suffix)
    if codec is None:
        raise ValueError("Unsupported local speech export format.")
    with av.open(str(destination), mode="w") as audio:
        stream = audio.add_stream(codec, rate=24000)
        frame = av.AudioFrame.from_ndarray(pcm[None, :], format="s16", layout="mono")
        frame.sample_rate = 24000
        for packet in stream.encode(frame):
            audio.mux(packet)
        for packet in stream.encode(None):
            audio.mux(packet)


if __name__ == "__main__":
    mode, incoming, outgoing = sys.argv[1:]
    {"stt": transcribe, "tts": synthesize}[mode](Path(incoming), Path(outgoing))
