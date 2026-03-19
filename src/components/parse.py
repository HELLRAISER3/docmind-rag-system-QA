from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader
)
from src.logging import logger


class Parse:
    LOADER_MAPPING = {
        ".pdf": PyMuPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
    }

    ALLOWED_METADATA_KEYS = {"source", "page", "file_path", "total_pages"}

    def __init__(self, 
                 file_folder: Path, 
                 clean_metadata: bool = True,
                 allowed_metadata_keys: set[str] = None):
        self.file_folder = Path(file_folder)
        self.clean_metadata = clean_metadata
        if allowed_metadata_keys:
            self.ALLOWED_METADATA_KEYS = allowed_metadata_keys
        
    def _clean_metadata(self, doc):
        doc.metadata = {
            k: v for k, v in doc.metadata.items() 
            if k in self.ALLOWED_METADATA_KEYS
        }
        return doc

    def parse_docs(self):
        all_documents = []
        
        if not self.file_folder.is_dir():
            raise ValueError(f"Path {self.file_folder} is not a directory.")

        for file_path in self.file_folder.iterdir():
            if file_path.is_dir():
                continue
                
            suffix = file_path.suffix.lower()
            
            if suffix in self.LOADER_MAPPING:
                
                logger.info(f"Parsing {suffix}: {file_path.name}")
                
                loader_class = self.LOADER_MAPPING[suffix]
                
                loader = loader_class(str(file_path))
                raw_docs = loader.load()

                if self.clean_metadata:
                    cleaned_docs = [self._clean_metadata(d) for d in raw_docs]
                    all_documents.extend(cleaned_docs)
                else: 
                    all_documents.extend(raw_docs)

            else:
                logger.info(f"Unsupported format: {suffix}")

        return all_documents