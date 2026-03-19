from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunk:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_doc(self, doc):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                  chunk_overlap=self.chunk_overlap)
        text_chunks = splitter.split_text(doc)
        return text_chunks
