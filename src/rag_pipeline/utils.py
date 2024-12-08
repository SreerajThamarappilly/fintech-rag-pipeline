# src/rag_pipeline/utils.py

def chunk_text(text: str, chunk_size: int):
    """
    Chunk a given text into smaller pieces of specified size.

    Args:
        text (str): The raw text.
        chunk_size (int): The maximum length of each chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for w in words:
        current_length += len(w) + 1
        if current_length > chunk_size:
            # Join current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [w]
            current_length = len(w) + 1
        else:
            current_chunk.append(w)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
