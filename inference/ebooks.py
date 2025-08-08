import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def epub_to_txt(epub_path, txt_path):
    book = epub.read_epub(epub_path)
    all_text = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            all_text.append(soup.get_text())
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

epub_to_txt("ebooks/Thus Spake Zarathustra.epub", "output.txt")
