import os

#import text_preprocess from src subdirectory
from src.text_preprocess import sentence_splitter, normalize

with open(os.path.join(os.path.dirname(__file__), 'pg11.txt'), 'r', encoding='utf-8') as file:
    text = file.read()

   
normalized_list = normalize(sentence_splitter(text))
print( normalized_list[0:10]) 

