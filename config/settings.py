from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_PATH = "data/imdb_movies.csv"
FAISS_INDEX_PATH = "faiss_index"
