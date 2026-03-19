from src.components.parse import Parse
from src.components.chunk import Chunk


parser = Parse("data", clean_metadata=True)
docs = parser.parse_docs()


chunker = Chunk(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_docs(docs)

print(chunks[:2])
