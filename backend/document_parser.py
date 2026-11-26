
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from docx import Document


def extract_text(file_bytes, filename):

    extension = Path(filename).suffix.lower()

    # Extract text from PDF
    if extension == ".pdf":

        pdf = PdfReader(BytesIO(file_bytes))

        text = ""

        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        return text

    # Extract text from DOCX
    elif extension == ".docx":

        document = Document(BytesIO(file_bytes))

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return text

    # Read plain text files
    elif extension == ".txt":

        return file_bytes.decode(
            "utf-8",
            errors="ignore"
        )

    else:
        raise ValueError(
            "Only PDF, DOCX and TXT files are supported"
        )
