import PyPDF2
from pathlib import Path

class FileHandler:
    """Handle file input/output operations"""

    SUPPORTED_FORMATS = {'.txt', '.pdf', '.docx'}

    def read_file(self, file_object) -> str:
        """Read content from uploaded file"""
        filename = file_object.name
        file_ext = Path(filename).suffix.lower()

        if file_ext == '.txt':
            return file_object.read().decode('utf-8', errors='ignore')

        elif file_ext == '.pdf':
            pdf_reader = PyPDF2.PdfReader(file_object)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @staticmethod
    def save_results(content: str, filename: str) -> None:
        """Save analysis results to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
