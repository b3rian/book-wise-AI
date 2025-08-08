
from ebooklib import epub
from bs4 import BeautifulSoup
import ebooklib

def epub_to_txt(epub_path, txt_path):
    book = epub.read_epub(epub_path)
    all_text = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text()
            # Split into lines and remove empty/whitespace-only lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            all_text.extend(lines)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

epub_to_txt("input.epub", "output.txt")