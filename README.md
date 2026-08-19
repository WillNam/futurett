# FutureTT

A self-developed, high-quality TTS voice cloning engine that reaches speeds of 150x realtime. Built on the ZipVoice architecture with custom 4-step flow-matching distillation and a 48 kHz vocoder.

## Features

- **Voice Cloning**: State-of-the-art voice cloning on par with models 10x larger
- **48 kHz Clarity**: Crystal-clear speech generation, unlike most TTS models limited to 24 kHz
- **150x Realtime**: Blazing fast on a single GPU, faster than realtime even on CPU
- **Lightweight**: Fits within 1 GB VRAM — runs on any local GPU, Apple MPS, or CPU
- **Multilingual**: Supports Chinese and English text input

## Install

```bash
git clone https://github.com/WillNam/futurett.git
cd futurett
pip install -r requirements.txt
```

## Usage

```python
from zipvoice.luxvoice import LuxTTS
import soundfile as sf

tts = LuxTTS(device='cuda')
encoded = tts.encode_prompt('reference_voice.wav', rms=0.01)
wav = tts.generate_speech("Hello, this is a test.", encoded, num_steps=4)
sf.write('output.wav', wav.numpy().squeeze(), 48000)
```

See [SKILL.md](SKILL.md) for full parameter reference and workflow.

## License

Apache-2.0
