import os
import torch
import torchaudio
import torchaudio.transforms as transforms
from tqdm import tqdm

#import text_preprocess from src subdirectory
from src.text_preprocess import TextProcessor
from src.text_to_speech import TextToSpeech
fname = "input/MiJ.txt"

processor = TextProcessor()
tts = TextToSpeech()
tts.model.prepare_conditionals("input/reference1.wav")

def check_spec(wav):
    sample_rate=24000
    transform = transforms.MelSpectrogram(sample_rate, n_fft=400, n_mels=128, hop_length=400)
    mel_specgram = transform(wav).squeeze(0)
    total_energy = mel_specgram[120:,-100:].sum(axis=0).sum()
    return total_energy.detach().cpu().numpy()


with open(os.path.join(os.path.dirname(__file__), fname), 'r', encoding='utf-8') as file:
    text = file.read()

f = open("output/MiJ/result.csv",'w')

# text = text[:2000]
paragraphs = processor.paragraph_splitter(text)
for idx, paragraph in enumerate(tqdm(paragraphs, desc="Paragraphs: ")):
    # Split each paragraph into manageable chunks for the TTS
    chunks = processor.sentence_splitter(paragraph, max_chars=400)
    normalized_list = processor.normalize(chunks)

    wavout = []
    for i, txt in enumerate(normalized_list):
        wav = tts.generate_speech(txt, exaggeration=.6, cfg_weight=0.4, temperature=1., repetition_penalty=1.0)
        diff = tts.check_tts(txt, wav)
        noise = check_spec(wav)
        f.write(f"{idx},{i},{diff},{noise}\n")
        wavout.append(wav)
    wavout=torch.hstack(wavout)
    torchaudio.save(f"output/MiJ/{idx}.wav", wavout, 24000)



    
