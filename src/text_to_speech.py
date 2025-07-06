from src.chatterbox.tts import ChatterboxTTS
import torch
import torchaudio

from typing import Optional
import whisper
import difflib
import re
import string
import traceback as tb
from whisper.normalizers import EnglishTextNormalizer

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
        self.stt_model = whisper.load_model("base.en", device=self.device)
        self.stt_options = whisper.DecodingOptions(language="en", without_timestamps=True)
        self.resampler = torchaudio.transforms.Resample(24_000, 16_000)
        self.normalizer = EnglishTextNormalizer()

    def prepare_conditionals(self, audio_prompt_path: str, exaggeration:float=0.5):
        """
        Prepares the conditionals for the TTS model.

        Args:
            audio_prompt_path (str): Path to an audio file to use as a voice prompt.
        """
        self.model.prepare_conditionals(audio_prompt_path, exaggeration)

    def generate_speech(self, text: str | list[str], audio_prompt_path: Optional[str] = None, exaggeration: float = 0.5,
        cfg_weight: float =0.5, temperature: float =0.8, repetition_penalty: float =1.0):
        """
        Generates speech from the given text.

        Args:
            text (str): The text to synthesize.
            audio_prompt_path (Optional[str]): Path to an audio file to use as a voice prompt.

        Returns:
            torch.Tensor: The generated audio waveform.
        """
        with torch.no_grad():
            chunk_generator = self.model.generate(text, audio_prompt_path=audio_prompt_path, 
                exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature, repetition_penalty=repetition_penalty)
            out = torch.cat(list(chunk_generator)).detach().cpu()
            # print(next(chunk_generator).shape)
        torch.cuda.synchronize()
        return out

    def check_tts(self, target_text: str, wav: torch.Tensor):

        def normalize_for_compare_all_punct(text):
            text = re.sub(r'[–—-]', ' ', text)
            text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
            text = re.sub(r'\s+', ' ', text)
            return text.lower().strip()

        try:
            audio = self.resampler(wav)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio.squeeze(0).to(self.device))
            result = self.stt_model.decode(mel, self.stt_options)
            transcribed = self.normalizer(result.text.strip().lower())
            target_text = self.normalizer(target_text.strip().lower())
            # print(f"\033[32m[DEBUG] Whisper transcription: '\033[33m{transcribed}'\033[0m")
            score = difflib.SequenceMatcher(
                None,
                normalize_for_compare_all_punct(transcribed),
                normalize_for_compare_all_punct(target_text)
            ).ratio()
            # print(f"\033[32m[DEBUG] Score: {score:.3f} (target: '\033[33m{target_text}')\033[0m")
            return (score)
        except Exception as e:
            tb.print_exc()
            print(f"[ERROR] Whisper transcription failed for {target_text}: {e}")
            return (0.0)

