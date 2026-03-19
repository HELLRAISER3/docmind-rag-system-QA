from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Chunk:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_doc(self, doc):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                  chunk_overlap=self.chunk_overlap)
        text_chunks = splitter.split_text(doc)
        return text_chunks
    
    def chunk_docs(self, docs: Document):
        chunks = []
        for doc in docs:
            chunked_doc = self._chunk_doc(doc.page_content)

            for i, text in enumerate(chunked_doc):
                chunks.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": doc.metadata.get("source", "unknown"),
                            "page": doc.metadata.get("page", 0),
                            "chunk_index": i,
                            "total_chunks": len(chunked_doc)
                        }
                    }
                )
        
        return chunks
