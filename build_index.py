import re
import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import OPENAI_API_KEY, DATA_PATH, FAISS_INDEX_PATH

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def split_star_cast(cast_string):
    """Split concatenated actor names by locating capital-letter boundaries."""
    capitals = [i for i, char in enumerate(cast_string) if char.isupper()]
    actors = []
    for i in range(0, len(capitals) - 1, 2):
        start = capitals[i]
        end = capitals[i + 2] if i + 2 < len(capitals) else len(cast_string)
        actor = cast_string[start:end].strip()
        actors.append(actor)
    return actors


def create_movie_description(row):
    return (
        f"{row['Title']} is a {row['Year']} {row['Genre']} film, "
        f"starring {row['Star Cast String']} and directed by {row['Director']}. "
        f"It has an IMDB Rating of {row['IMDb Rating']}."
    )


def prepare_documents(dataframe):
    documents = []
    for _, row in dataframe.iterrows():
        metadata = {
            "title": row["Title"],
            "genre": row["Genre"],
            "year": int(row["Year"]),
            "director": row["Director"],
            "rating": float(row["IMDb Rating"]),
            "cast": row["Star Cast String"],
            "poster": row["Poster-src"],
        }
        doc = Document(page_content=row["description"], metadata=metadata)
        documents.append(doc)
    return documents


def main():
    print(f"Loading dataset from {DATA_PATH}...")
    movie_data = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(movie_data)} movies.")

    print("Cleaning Star Cast field...")
    movie_data["Star Cast Cleaned"] = movie_data["Star Cast"].apply(split_star_cast)
    movie_data["Star Cast String"] = movie_data["Star Cast Cleaned"].apply(
        lambda x: ", ".join(x)
    )

    print("Building movie descriptions...")
    movie_data["description"] = movie_data.apply(create_movie_description, axis=1)

    print("Converting to LangChain documents...")
    documents = prepare_documents(movie_data)
    print(f"Created {len(documents)} documents.")

    print("Creating OpenAI embeddings model...")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Building FAISS vector store (this may take 1-2 minutes)...")
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings_model)
    print(f"FAISS index built — {vectorstore.index.ntotal} vectors stored.")

    print(f"Saving index to {FAISS_INDEX_PATH}/...")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("Done! FAISS index saved successfully.")


if __name__ == "__main__":
    main()
