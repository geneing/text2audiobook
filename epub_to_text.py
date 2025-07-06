import os
import sys

import argparse
import numpy as np
import re
import subprocess
import torch
import warnings
from tqdm import tqdm

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

from lxml import etree
from PIL import Image
import zipfile
import warnings

warnings.filterwarnings("ignore")

namespaces = {
   "calibre":"http://calibre.kovidgoyal.net/2009/metadata",
   "dc":"http://purl.org/dc/elements/1.1/",
   "dcterms":"http://purl.org/dc/terms/",
   "opf":"http://www.idpf.org/2007/opf",
   "u":"urn:oasis:names:tc:opendocument:xmlns:container",
   "xsi":"http://www.w3.org/2001/XMLSchema-instance",
}

warnings.filterwarnings("ignore", module="ebooklib.epub")


def conditional_sentence_case(sent):
    # Split the sentence into words
    words = sent.split()

    # Check if the first two words are uppercase
    if len(words) >= 2 and words[0].isupper() and words[1].isupper():
        # Convert the entire sentence to lowercase, then capitalize the first letter
        sent = sent.lower().capitalize()

    return sent

def chap2text_epub(chap):
    blacklist = [
        "[document]",
        "noscript",
        "header",
        "html",
        "meta",
        "head",
        "input",
        "script",
    ]
    paragraphs = []
    soup = BeautifulSoup(chap, "html.parser")

    # Extract chapter title (assuming it's in an <h1> tag)
    chapter_title = soup.find("h1")
    if chapter_title:
        chapter_title_text = chapter_title.text.strip()
    else:
        chapter_title_text = None

    # Always skip reading links that are just a number (footnotes)
    for a in soup.findAll("a", href=True):
        if not any(char.isalpha() for char in a.text):
            a.extract()

    chapter_paragraphs = soup.find_all("p")
    if len(chapter_paragraphs) == 0:
        print(f"Could not find any paragraph tags <p> in \"{chapter_title_text}\". Trying with <div>.")
        chapter_paragraphs = soup.find_all("div")

    for p in chapter_paragraphs:
        paragraph_text = "".join(p.strings).strip()
        paragraphs.append(paragraph_text)

    return chapter_title_text, paragraphs

def get_epub_cover(epub_path):
    try:
        with zipfile.ZipFile(epub_path) as z:
            t = etree.fromstring(z.read("META-INF/container.xml"))
            rootfile_path =  t.xpath("/u:container/u:rootfiles/u:rootfile",
                                        namespaces=namespaces)[0].get("full-path")

            t = etree.fromstring(z.read(rootfile_path))
            cover_meta = t.xpath("//opf:metadata/opf:meta[@name='cover']",
                                        namespaces=namespaces)
            if not cover_meta:
                print("No cover image found.")
                return None
            cover_id = cover_meta[0].get("content")

            cover_item = t.xpath("//opf:manifest/opf:item[@id='" + cover_id + "']",
                                            namespaces=namespaces)
            if not cover_item:
                print("No cover image found.")
                return None
            cover_href = cover_item[0].get("href")
            cover_path = os.path.join(os.path.dirname(rootfile_path), cover_href)
            if os.name == 'nt' and '\\' in cover_path:
                cover_path = cover_path.replace("\\", "/")
            return z.open(cover_path)
    except FileNotFoundError:
        print(f"Could not get cover image of {epub_path}")

def export(book, sourcefile):
    book_contents = []
    cover_image = get_epub_cover(sourcefile)
    image_path = None

    if cover_image is not None:
        image = Image.open(cover_image)
        image_filename = sourcefile.replace(".epub", ".png")
        image_path = os.path.join(image_filename)
        image.save(image_path)
        print(f"Cover image saved to {image_path}")

    spine_ids = []
    for spine_tuple in book.spine:
        if spine_tuple[1] == 'yes': # if item in spine is linear
            spine_ids.append(spine_tuple[0])

    items = {}
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            items[item.get_id()] = item

    for id in spine_ids:
        item = items.get(id, None)
        if item is None:
            continue
        chapter_title, chapter_paragraphs = chap2text_epub(item.get_content())
        book_contents.append({"title": chapter_title, "paragraphs": chapter_paragraphs})
    outfile = sourcefile.replace(".epub", ".txt")
    
    print(f"Exporting {sourcefile} to {outfile}")
    author = book.get_metadata("DC", "creator")[0][0]
    booktitle = book.get_metadata("DC", "title")[0][0]

    with open(outfile, "w", encoding='utf-8') as file:
        #ei 
        # file.write(f"Title: {booktitle}\n")
        # file.write(f"Author: {author}\n\n")

        # file.write(f"# Title\n")
        # file.write(f"{booktitle}, by {author}\n\n")
        for i, chapter in enumerate(book_contents, start=1):
            if chapter["paragraphs"] == [] or chapter["paragraphs"] == ['']:
                continue
            else:
                if chapter["title"] == None:
                    # file.write(f"# Part {i}\n")
                    pass
                else:
                    file.write(f"# {chapter['title']}\n\n")
                for paragraph in chapter["paragraphs"]:
                    clean = re.sub(r'[\s\n]+', ' ', paragraph)
                    clean = re.sub(r'[“”]', '"', clean)  # Curly double quotes to standard double quotes
                    clean = re.sub(r'[‘’]', "'", clean)  # Curly single quotes to standard single quotes
                    clean = re.sub(r'--', ', ', clean)
                    file.write(f"{clean}\n\n")

def get_book(sourcefile):
    book_contents = []
    book_title = sourcefile
    book_author = "Unknown"
    chapter_titles = []

    with open(sourcefile, "r", encoding="utf-8") as file:
        current_chapter = {"title": "blank", "paragraphs": []}
        initialized_first_chapter = False
        lines_skipped = 0
        for line in file:

            if lines_skipped < 2 and (line.startswith("Title") or line.startswith("Author")):
                lines_skipped += 1
                if line.startswith('Title: '):
                    book_title = line.replace('Title: ', '').strip()
                elif line.startswith('Author: '):
                    book_author = line.replace('Author: ', '').strip()
                continue

            line = line.strip()
            if line.startswith("#"):
                if current_chapter["paragraphs"] or not initialized_first_chapter:
                    if initialized_first_chapter:
                        book_contents.append(current_chapter)
                    current_chapter = {"title": None, "paragraphs": []}
                    initialized_first_chapter = True
                chapter_title = line[1:].strip()
                if any(c.isalnum() for c in chapter_title):
                    current_chapter["title"] = chapter_title
                    chapter_titles.append(current_chapter["title"])
                else:
                    current_chapter["title"] = "blank"
                    chapter_titles.append("blank")
            elif line:
                if not initialized_first_chapter:
                    chapter_titles.append("blank")
                    initialized_first_chapter = True
                if any(char.isalnum() for char in line):
                    sentences = sent_tokenize(line)
                    cleaned_sentences = [s for s in sentences if any(char.isalnum() for char in s)]
                    line = ' '.join(cleaned_sentences)
                    current_chapter["paragraphs"].append(line)

        # Append the last chapter if it contains any paragraphs.
        if current_chapter["paragraphs"]:
            book_contents.append(current_chapter)

    return book_contents, book_title, book_author, chapter_titles

def sort_key(s):
    # extract number from the string
    return int(re.findall(r'\d+', s)[0])

def fix_sentence_length(sentences):
    fixed_sentences = []
    skip_next = False
    for i, sentence in enumerate(sentences):
        if skip_next:
            skip_next = False
            continue
        words = sentence.split()
        if len(words) < 3:
            if i < len(sentences) - 1:
                # Combine with next sentence
                next_sentence = sentences[i + 1]
                combined_sentence = sentence + " " + next_sentence
                fixed_sentences.append(combined_sentence)
                skip_next = True  # Skip the next sentence as it's already combined
            else:
                # Last sentence, combine with previous sentence
                if fixed_sentences:
                    fixed_sentences[-1] += " " + sentence
                else:
                    fixed_sentences.append(sentence)
        else:
            fixed_sentences.append(sentence)
    return fixed_sentences

def read_book(book_contents, sample, notitles):
    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    current_device = torch.device(device)
    print(f"Attempting to use device: {device}")
    model = ChatterboxTTS.from_pretrained(device=device)

    segments = []
    for i, chapter in enumerate(book_contents, start=1):
        paragraphpause = 600  # default pause between paragraphs in ms
        files = []
        partname = f"part{i}.flac"
        print(f"\n\n")

        if os.path.isfile(partname):
            print(f"{partname} exists, skipping to next chapter")
            segments.append(partname)
        else:
            print(f"Chapter ({i}/{len(book_contents)}): {chapter['title']}\n")
            print(f"Section name: \"{chapter['title']}\"")
            if chapter["title"] == "":
                chapter["title"] = "blank"
            if chapter["title"] != "Title" and notitles != True:
                chapter['paragraphs'][0] = chapter['title'] + ". " + chapter['paragraphs'][0]
            for pindex, paragraph in enumerate(chapter["paragraphs"]):
                ptemp = f"pgraphs{pindex}.flac"
                if os.path.isfile(ptemp):
                    print(f"{ptemp} exists, skipping to next paragraph")
                else:
                    sentences = sent_tokenize(paragraph)
                    # This is probably not needed, commenting out for now
                    # sentences = fix_sentence_length(sentences)
                    filenames = [
                        "sntnc" + str(z) + ".wav" for z in range(len(sentences))
                    ]
                    chatterbox_read(sentences, sample, filenames, model)
                    append_silence(filenames[-1], paragraphpause)
                    # combine sentences in paragraph
                    sorted_files = sorted(filenames, key=sort_key)
                    #if os.path.exists("sntnc0.wav"):
                    #    sorted_files.insert(0, "sntnc0.wav")
                    combined = AudioSegment.empty()
                    for file in sorted_files:
                        # try/except prob not needed, actual failure was from a torch recursive error that was fixed
                        try:
                            combined += AudioSegment.from_file(file)
                        except:
                            print("FAILURE at sorted file combine")
                            print(f"File: {file}")
                            print(f"sorted files: {sorted_files}")
                            print(f"Unsorted: {filenames}")
                            sys.exit()
                    combined.export(ptemp, format="flac")
                    for file in sorted_files:
                        os.remove(file)
                files.append(ptemp)
            # combine paragraphs into chapter
            append_silence(files[-1], 2000)
            combined = AudioSegment.empty()
            for file in files:
                combined += AudioSegment.from_file(file)
            combined.export(partname, format="flac")
            for file in files:
                os.remove(file)
            segments.append(partname)
    return segments
    
epub_filename = "input/pg103-images-3.epub"
book = epub.read_epub(epub_filename)
export(book, epub_filename)
# book_contents, book_title, book_author, chapter_titles = get_book(epub_filename)
print(book)