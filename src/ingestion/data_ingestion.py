import os
from dotenv import load_dotenv
from pathlib import Path
import logging
import json
import datetime
import time

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


from src.ingestion.chunking_utility import repair_and_filter_chunks
from src.ingestion.get_movies_from_db import GetMoviesFromDB
from src import get_vector_store
from src.utils import logError

from src.config import APP_MODE, FINGERPRINT
from src.fingerprint import fingerprint_matches, write_fingerprint
from src.document_deserialization import json_to_documents

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataIngestion:

    def __init__(self, batch_size, last_processed_id=0):
        load_dotenv()
        self.batch_size = batch_size
        self.last_processed_id = last_processed_id

    def ingest_from_json(self):
        stage = "start"
        logger.info("[START] INGESTION")
        DEMO_DOCS_PATH = os.getenv("DEMO_DOCS", "data/processed/demo_chunks.json")

        try:
            vector_store = get_vector_store()
            persist_dir = os.getenv("CHROMA_DIR_DEMO", "data/chroma_demo/v0")

            # Check fingerprint validity
            is_valid = os.path.exists(persist_dir) and fingerprint_matches(
                persist_dir, FINGERPRINT
            )

            print("is valid? ", is_valid)
            print("finger: ", FINGERPRINT)

            if is_valid:
                return vector_store

            with open(DEMO_DOCS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)

            documents = json_to_documents(raw)
            vector_store.add_documents(documents)
            write_fingerprint(persist_dir, FINGERPRINT)
            logger.info("Documents added in vector store")

        except Exception as e:
            logger.error(
                "Error in data ingestion from json at stage '%s': %s",
                stage,
                e,
                exc_info=True,
            )
            logError(
                e,
                "DataIngestion.ingest_from_json",
                f"Error in data ingestion at stage {stage}",
            )

    def ingest(self):
        stage = "start"
        logger.info("[START] INGESTION")

        try:
            vector_store = get_vector_store()
            movie_details_instance = GetMoviesFromDB()
            conn = movie_details_instance.get_db_connection()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " "],
                add_start_index=True,
            )

            i = 0
            movie_count = 0

            for batched_movie_details in movie_details_instance.get_batched_details(
                self.batch_size, conn
            ):
                movie_ids = list()
                batched_docs = list()

                logger.info("Batch: %s", i)
                if batched_movie_details is None or len(batched_movie_details) == 0:
                    continue

                stage = "documenting"
                logger.info("Documenting starts")

                for movie in batched_movie_details:
                    metadata = {
                        "id": movie["id"],
                        "tmdb_id": movie["tmdb_id"],
                        "title": movie["title"],
                        "alt_title": movie["alt_title"],
                        "release_year": movie["release_year"],
                        "original_language": movie["original_language"],
                        "adult": movie["adult"],
                        "tagline": movie["tagline"],
                        "book_adaptation": movie["book_adaptation"],
                        "genres": movie["genres"],
                        "actors": movie["actors"],
                        "directors": movie["directors"],
                        "producers": movie["producers"],
                    }

                    documents = []

                    # Tagline
                    if movie.get("tagline"):
                        documents.append(
                            Document(
                                page_content=f"Tagline: {movie['tagline']}",
                                metadata={
                                    **metadata,
                                    "chunkType": "tagline",
                                    "chunkIndex": 0,
                                },
                            )
                        )

                    # Short plot
                    if movie.get("plot_short"):
                        documents.append(
                            Document(
                                page_content=f"Short Plot: {movie['plot_short']}",
                                metadata={
                                    **metadata,
                                    "chunkType": "short",
                                    "chunkIndex": 0,
                                },
                            )
                        )

                    # Detailed plot (will be chunked)
                    if movie.get("plot_detailed"):
                        documents.append(
                            Document(
                                page_content=f"{movie['plot_detailed']}",
                                metadata={
                                    **metadata,
                                    "chunkType": "detailed",
                                    "chunkIndex": 0,
                                },
                            )
                        )

                    # Title document
                    documents.append(
                        Document(
                            page_content=f"Movie title: {movie['alt_title']}. Genres: {movie.get('genres', '')}",
                            metadata={
                                **metadata,
                                "chunkType": "title",
                                "chunkIndex": 0,
                            },
                        )
                    )

                    batched_docs.extend(documents)
                    movie_ids.append(movie["tmdb_id"])
                    movie_count += 1

                stage = "Chunking"
                logger.info("Chunking starts")

                chunking_start_time = time.time()

                # Only chunk detailed plots
                plot_docs = [
                    d for d in batched_docs if d.metadata["chunkType"] == "detailed"
                ]
                other_docs = [
                    d for d in batched_docs if d.metadata["chunkType"] != "detailed"
                ]

                raw_chunks = text_splitter.split_documents(plot_docs)

                plot_chunks = repair_and_filter_chunks(
                    raw_chunks,
                    max_chars=1200,
                    min_chars=40,
                )

                # Update chunk indices
                for idx, chunk in enumerate(plot_chunks):
                    chunk.metadata["chunkIndex"] = (
                        f"{chunk.metadata['alt_title']}_{idx}"
                    )

                chunked_docs = plot_chunks + other_docs

                chunking_end_time = time.time()

                time_elapsed_chunking = f"{(chunking_end_time-chunking_start_time):.2f}"
                logger.info("Time to chunk:  %ss", time_elapsed_chunking)
                stage = "embedding_and_storing"
                embedding_start_time = time.time()

                vector_store.add_documents(chunked_docs)

                embedding_end_time = time.time()

                time_elapsed_embedding = (
                    f"{(embedding_end_time-embedding_start_time):.2f}"
                )
                logger.info("Time to embed and store:  %ss", time_elapsed_embedding)

                # Log processing
                dataToStore = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "batch": i,
                    "ids": movie_ids,
                }

                log_dir = Path(f"logs/{datetime.date.today().strftime('%Y_%m_%d')}")
                log_dir.mkdir(parents=True, exist_ok=True)

                with open(
                    log_dir
                    / f"ingestion_logs_{datetime.datetime.now().strftime('%Y_%m_%d_%H')}.json",
                    "a",
                ) as f:
                    json.dump(dataToStore, f)
                    f.write("\n")

                i += 1
                logger.info("Ingestion complete for batch %s", i)
                print("-" * 90)

            logger.info("✅ Data ingestion complete for %s movies", movie_count)

        except Exception as e:
            logger.error(
                "Error in data ingestion at stage '%s': %s", stage, e, exc_info=True
            )
            logError(
                e, "DataIngestion.ingest", f"Error in data ingestion at stage {stage}"
            )


def main():
    di = DataIngestion(batch_size=200, last_processed_id=0)
    # di.ingest()
    if APP_MODE == "demo":
        di.ingest_from_json()
    else:
        di.ingest()


if __name__ == "__main__":
    main()
