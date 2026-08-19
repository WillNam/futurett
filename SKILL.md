---
name: futurett
description: "Self-developed high-quality rapid TTS voice cloning engine. Encode a reference audio clip and generate cloned speech at 150x realtime, 48 kHz. Use when the user asks for 语音克隆, voice cloning, TTS, 文字转语音, or wants to generate speech that sounds like a specific person."
---

# FutureTT — Voice Cloning Engine

A self-developed, high-quality voice cloning and text-to-speech engine built on the ZipVoice architecture with custom 4-step flow-matching distillation and a proprietary 48 kHz vocoder.

## Capabilities

- **Voice cloning**: SOTA quality on par with models 10x larger
- **48 kHz output**: Clear speech (most TTS models are limited to 24 kHz)
- **150x realtime** on GPU; faster-than-realtime on CPU
- **< 1 GB VRAM**: Runs on any local GPU, MPS (Mac), or CPU

## Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchaudio`, `vocos`, `librosa`, `safetensors`, `huggingface_hub`, `piper-phonemize`, `linacodec` (from git).

Model weights are downloaded automatically on first use.

## Quick Start

```python
from zipvoice.luxvoice import LuxTTS
import soundfile as sf

# Load model (auto-detects cuda → mps → cpu)
tts = LuxTTS(device='cuda')

# Encode a reference voice (min 3 seconds, wav/mp3)
encoded = tts.encode_prompt('reference_voice.wav', rms=0.01)

# Generate speech
wav = tts.generate_speech(
    "大家好，我是 Conan。今天我们聊一个很有意思的话题。",
    encoded,
    num_steps=4,
)

sf.write('output.wav', wav.numpy().squeeze(), 48000)
```

## Parameters

### `encode_prompt(audio_path, duration=5, rms=0.001)`

| Param | Default | Description |
|---|---|---|
| `duration` | 5 | Reference audio duration in seconds. Lower = faster. Set to 1000 if artifacts appear. |
| `rms` | 0.001 | Volume normalization. 0.01 recommended for louder output. |

### `generate_speech(text, encoded, ...)`

| Param | Default | Description |
|---|---|---|
| `num_steps` | 4 | Diffusion steps. 3–4 best for speed/quality balance. |
| `t_shift` | 0.5 | Sampling temperature. Higher = better quality but possibly worse WER. 0.9 recommended. |
| `speed` | 1.0 | Playback speed. Lower = slower speech. |
| `return_smooth` | False | Smoother output (24 kHz) but less crisp. |
| `guidance_scale` | 3.0 | Classifier-free guidance scale. |

## Workflow

1. **Prepare reference audio**: Use a clean 3–30 second clip of the target voice (no background music/noise).
2. **Encode once**: `encode_prompt()` — reuse the encoded dict for multiple generations.
3. **Generate**: `generate_speech()` with your text. Supports Chinese and English.
4. **Save**: Output is 48 kHz WAV via `soundfile`.

## Output Structure

```
outputs/futurett/
  reference/          # Reference voice clips
  generated/          # Generated audio files
  MANIFEST.md         # Generation log (text, params, output path)
```

## Tips

- Use at minimum a 3-second reference audio file.
- Set `return_smooth=True` if you hear metallic artifacts.
- Lower `t_shift` for fewer pronunciation errors (at the cost of naturalness).
- For Mac users, the model auto-falls back to MPS if CUDA is unavailable.

## Architecture

FutureTT uses a ZipVoice backbone distilled to 4 diffusion steps via custom flow-matching, paired with a 48 kHz Vocos-based vocoder (vs. the standard 24 kHz). The result is a sub-1GB model delivering studio-quality cloned speech at 150x realtime.

## License

Apache-2.0
