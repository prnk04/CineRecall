import sqlite3
import logging
import os
from dotenv import load_dotenv

from src.utils import logError

logger = logging.getLogger(__name__)


class DatabaseOperations:
    def __init__(self):
        load_dotenv()
        self.db_file_path = os.getenv("SQLITE_DB_PATH", "")

    def get_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def flush_batch(
        self,
        conn,
        movies_batch,
        crew_batch,
        roles_batch,
        plots_batch,
        genres_batch,
        movie_genre_batch,
        movie_mapping_batch,
    ):
        stage = None
        data = None

        try:
            with conn:
                stage = "MOVIE_DETAILS"
                data = movies_batch
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO MOVIE_DETAILS
                    ( tmdb_id, imdb_id, title, release_year, original_language, adult, tagline, book_adaptation, created_at)
                    VALUES (?,?,?,?,?,?,?,?, ?)
                    """,
                    movies_batch,
                )

                stage = "CAST_CREW_DETAILS"
                data = crew_batch
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO CAST_CREW_DETAILS
                    ( name, tmdb_id, created_at)
                    VALUES (?,?,?)
                    """,
                    crew_batch,
                )

                stage = "MOVIE_CREW_DETAILS"
                data = roles_batch
                conn.executemany(
                    """
                    INSERT INTO MOVIE_CREW_DETAILS
                    (movie_id, crew_id, character, job, description, created_at)
                    VALUES (?,?,?,?,?,?)
                    """,
                    roles_batch,
                )

                stage = "MOVIE_PLOT"
                data = plots_batch
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO MOVIE_PLOT
                    (movie_id, plot_type, plot, created_at)
                    VALUES (?,?,?, ?)
                    """,
                    plots_batch,
                )

                stage = "GENRES"
                data = genres_batch
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO GENRES 
                    (genre, created_at) 
                    VALUES (?, ?)
                    """,
                    genres_batch,
                )

                stage = "CAST_CREW_DETAILS"
                data = movie_genre_batch

                genre_id_mapping = dict()
                genre_names = set()
                for genre_det in movie_genre_batch:
                    this_genre_name = genre_det[1]
                    genre_names.add(this_genre_name)

                # first, check the ids of these genres in db

                placeholders = ",".join("?" for _ in genre_names)

                query = f"""
                    SELECT id, genre
                    FROM GENRES
                    WHERE genre IN ({placeholders})
                """

                cursor = conn.execute(query, tuple(genre_names))
                rows = cursor.fetchall()

                genre_id_mapping = {name: genre_id for genre_id, name in rows}

                new_list = list()
                for old_data in movie_genre_batch:
                    updated_data = [
                        old_data[0],
                        genre_id_mapping.get(old_data[1]),
                        old_data[2],
                    ]
                    new_list.append(updated_data)

                data = new_list

                conn.executemany(
                    """
                    INSERT OR IGNORE INTO MOVIE_GENRE_DETAILS
                    ( movie_id, genre_id, created_at)
                    VALUES (?,?, ?)
                    """,
                    new_list,
                )
                stage = "MOVIE_MAPPING"
                data = movie_mapping_batch
                conn.executemany(
                    """
                INSERT OR IGNORE INTO MOVIE_NAME_MAPPING
                ("tmdb_id", "original_title", "alt_title")
                VALUES (?,?,?)
                """,
                    data,
                )
        except Exception as e:
            print("Exception occurred in inserting the data: ", e)
            logger.exception(
                "DB insertion failed at stage =%s | batch_size = %s", stage, data
            )
            logError(
                e,
                "DatabaseOperations.flush_batch",
                f"DB insertion failed at stage {stage} and batc size: {data}",
            )
            raise RuntimeError("Batch insert failed") from e
        # finally:
