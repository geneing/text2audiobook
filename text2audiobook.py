import os

#import text_preprocess from src subdirectory
from src.text_preprocess import TextProcessor
from src.text_to_speech import TextToSpeech

with open(os.path.join(os.path.dirname(__file__), 'pg11.txt'), 'r', encoding='utf-8') as file:
    text = file.read()

text = text[0:1000]
processor = TextProcessor()
sentences = processor.sentence_splitter(text)
normalized_list = processor.normalize(sentences)
print(normalized_list)

tts = TextToSpeech()
wav = tts.generate_speech(normalized_list[0], audio_prompt_path="1.wav")

print(next(wav).shape)