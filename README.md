# FutureTT

High-quality rapid TTS voice cloning skill based on [LuxTTS](https://github.com/ysharma3501/LuxTTS).

- Voice cloning at 150x realtime, 48 kHz output
- < 1 GB VRAM — runs on GPU, MPS (Mac), or CPU
- Supports Chinese and English

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

tts = LuxTTS('YatharthS/LuxTTS', device='cuda')
encoded = tts.encode_prompt('reference_voice.wav', rms=0.01)
wav = tts.generate_speech("Hello, this is a test.", encoded, num_steps=4)
sf.write('output.wav', wav.numpy().squeeze(), 48000)
```

See [SKILL.md](SKILL.md) for full parameter reference and workflow.

## License

Apache-2.0
