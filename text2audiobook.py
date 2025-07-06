import os
import torch
import torchaudio
import pickle

#import text_preprocess from src subdirectory
from src.text_preprocess import TextProcessor
from src.text_to_speech import TextToSpeech
fname = "input/pg103-images-3.txt"
pkl_name = fname.replace(".txt", ".pkl")

processor = TextProcessor()
tts = TextToSpeech()
tts.model.prepare_conditionals("input/1.wav")

with open(os.path.join(os.path.dirname(__file__), fname), 'r', encoding='utf-8') as file:
    text = file.read()

# text = text[:2000]
paragraphs = processor.paragraph_splitter(text)
for idx, paragraph in enumerate(paragraphs):
    # Split each paragraph into manageable chunks for the TTS
    chunks = processor.sentence_splitter(paragraph, max_chars=400)
    normalized_list = processor.normalize(chunks)

    wavout = []
    for i, txt in enumerate(normalized_list):
        wav = tts.generate_speech(txt, exaggeration=.5, cfg_weight=0.5, temperature=1., repetition_penalty=1.0)
        diff = tts.check_tts(txt, wav)
        # print(f"{wav.shape} {type(wav)}")
        wavout.append(wav)
    wavout=torch.hstack(wavout)
    torchaudio.save(f"output/out_{idx}.wav", wavout, 24000)
    if idx > 3: break


