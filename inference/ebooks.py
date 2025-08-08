
from ebooklib import epub
from bs4 import BeautifulSoup
import ebooklib
import os

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

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each EPUB file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.epub'):
            epub_path = os.path.join(input_folder, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_filename)
            
            print(f"Processing: {filename} -> {txt_filename}")
            epub_to_txt(epub_path, txt_path)
    
    print("Conversion complete!")

# Example usage:
input_folder = "D:/Documents/Nietzsche"  # Folder containing your EPUB files
output_folder = "D:/Documents/text_files"  # Folder where TXT files will be saved
process_folder(input_folder, output_folder)