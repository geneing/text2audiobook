from src.chatterbox.tts import ChatterboxTTS
import torch
from typing import Optional

def _t3_to(model: "ChatterboxTTS", dtype):
    model.t3.to(dtype=dtype)
    model.t3.cond_enc.spkr_enc.to(dtype=dtype)
    model.conds.t3.to(dtype=dtype)
    return model

def _compile_t3(model: ChatterboxTTS):
    model.t3._step_compilation_target_original = model.t3._step_compilation_target
    model.t3._step_compilation_target = torch.compile(model.t3._step_compilation_target, fullgraph=True, backend="cudagraphs")
    return model

def _remove_t3_compilation(model: ChatterboxTTS):
    model.t3._step_compilation_target = model.t3._step_compilation_target_original
    return model

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
        #ei debug
        _t3_to(self.model, torch.bfloat16)
        self.model = _compile_t3(self.model)

    def prepare_conditionals(self, audio_prompt_path: str, exaggeration:float=0.5):
        """
        Prepares the conditionals for the TTS model.

        Args:
            audio_prompt_path (str): Path to an audio file to use as a voice prompt.
        """
        self.model.prepare_conditionals(audio_prompt_path, exaggeration)

    def generate_speech(self, text: str | list[str], audio_prompt_path: Optional[str] = None, exaggeration: float = 0.5,
        cfg_weight: float =0.5, temperature: float =0.8,):
        """
        Generates speech from the given text.

        Args:
            text (str): The text to synthesize.
            audio_prompt_path (Optional[str]): Path to an audio file to use as a voice prompt.

        Returns:
            torch.Tensor: The generated audio waveform.
        """
        with torch.no_grad():
            chunk_generator = self.model.generate(text, audio_prompt_path=audio_prompt_path, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
            out = torch.cat(list(chunk_generator)).detach().cpu()
            # print(next(chunk_generator).shape)



        torch.cuda.synchronize()
        return out

