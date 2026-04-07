import os
os.environ["SUNO_OFFLOAD_CPU"] = "1"  # Offload to CPU when not in use
os.environ["SUNO_USE_SMALL_MODELS"] = "1"  # Use smaller model variants
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav

torch.serialization.add_safe_globals([np.core.multiarray.scalar])
from bark import SAMPLE_RATE, generate_audio, preload_models

preload_models()

text_prompt = """
    Create rock and roll rhythm riff in the key of B minor, with a dark, suspenseful tone.
"""

audio_array = generate_audio(text_prompt)
write_wav("bark.wav", SAMPLE_RATE, audio_array)
