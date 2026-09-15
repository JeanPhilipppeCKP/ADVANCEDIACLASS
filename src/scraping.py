import trafilatura
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path

OUTPUT_FILE = Path("data/processed/corpus.txt")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def ocr_pdf_page(pdf_path: str, page_number: int) -> str:
    """Convertit une page PDF en image puis applique l'OCR dessus."""
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    return pytesseract.image_to_string(images[0], lang="eng")


def extract_pdf_text(pdf_path: str) -> str:
    """Extrait le texte d'un PDF, page par page, avec repères de page."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""

            if not page_text.strip():
                print(f"Page {i} sans texte détecté, passage en OCR...")
                page_text = ocr_pdf_page(pdf_path, i)

            text_parts.append(f"\n--- Page {i} ---\n{page_text}")
    return "\n".join(text_parts)


def extract_web_text(url: str) -> str:
    """Extrait le contenu principal d'une page web (hors nav/pub/footer)."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError(f"Impossible de récupérer {url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    return text or ""


def append_to_corpus(content: str, source_name: str):
    """Ajoute le contenu au fichier corpus unique, avec un en-tête de source."""
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n\n===== SOURCE: {source_name} =====\n\n")
        f.write(content)
    print(f"Ajouté au corpus : {source_name}")


if __name__ == "__main__":
    pdf_text = extract_pdf_text("data/raw/GpipePDF.pdf")
    append_to_corpus(pdf_text, "GpipePDF.pdf")

    pdf_text = extract_pdf_text("data/raw/matan-90.pdf")
    append_to_corpus(pdf_text, "matan-90.pdf")

    web_text = extract_web_text("https://www.iguazio.com/glossary/llm-embeddings/")
    append_to_corpus(web_text, "iguazio_llm_embeddings")