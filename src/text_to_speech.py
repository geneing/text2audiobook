from src.chatterbox.tts import ChatterboxTTS
import torch
from typing import Optional


class TextToSpeech:
    """
    A class for performing Text-to-Speech using ChatterboxTTS.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initializes the TextToSpeech class and loads the ChatterboxTTS model.

        Args:
            device (Optional[str]): The device to use for inference (e.g., "cuda", "mps", "cpu").
                                    If None, the best available device will be automatically detected.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")
        self.model = ChatterboxTTS.from_pretrained(device=self.device)

    def generate_speech(self, text: str, audio_prompt_path: Optional[str] = None):
        """
        Generates speech from the given text.

        Args:
            text (str): The text to synthesize.
            audio_prompt_path (Optional[str]): Path to an audio file to use as a voice prompt.

        Returns:
            torch.Tensor: The generated audio waveform.
        """
        return self.model.generate(text, audio_prompt_path=audio_prompt_path)
