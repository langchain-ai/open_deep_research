class PdfReadError(RuntimeError):
    pass


def clean_pdf_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())
