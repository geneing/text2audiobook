import os
import torch
import torchaudio
import pickle

#import text_preprocess from src subdirectory
from src.text_preprocess import TextProcessor
from src.text_to_speech import TextToSpeech
fname = "input/pg103-images-3.txt"
pkl_name = fname.replace(".txt", ".pkl")

if True: #not os.path.exists(pkl_name):
    with open(os.path.join(os.path.dirname(__file__), fname), 'r', encoding='utf-8') as file:
        text = file.read()

    processor = TextProcessor()
    text = text[:2000]
    sentences = processor.sentence_splitter(text, max_chars=400)
    print(sentences)
    normalized_list = processor.normalize(sentences)
    # print(normalized_list)
    with open(os.path.join(os.path.dirname(__file__), pkl_name), 'wb') as file:
        pickle.dump(normalized_list, file)

exit(0)

tts = TextToSpeech()
tts.model.prepare_conditionals("input/1.wav")
wavout = []
for i, txt in enumerate(normalized_list):
    wav = tts.generate_speech(txt, exaggeration=.7, cfg_weight=0.3, temperature=0.8)
    # print(f"{wav.shape} {type(wav)}")
    wavout.append(wav)
    # torchaudio.save(f"output/out_{i}.wav", wav, 24000)

wavout=torch.hstack(wavout)
print(f"{wavout.shape=}")
torchaudio.save("output/out.wav", wavout, 24000)

