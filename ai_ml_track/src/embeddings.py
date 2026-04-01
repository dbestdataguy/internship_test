from sentence_transformers import SentenceTransformer

# Load the model once when this module is imported.
# all-MiniLM-L6-v2 is a lightweight model that runs on CPU,
# produces 384-dimensional vectors, and is very good at
# capturing semantic similarity between short texts.
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def get_embeddings(descriptions: list[str]):
    """
    Convert a list of transaction descriptions into
    a 2D array of vectors.

    Input:
        descriptions: ["Uber trip 1200", "Netflix subscription 4500", ...]

    Output:
        A numpy array of shape (16, 384)
        i.e. 16 rows, each row is 384 numbers
    """
    if not descriptions:
        raise ValueError("Cannot embed an empty list of descriptions.")

    embeddings = model.encode(descriptions, show_progress_bar=True)
    return embeddings