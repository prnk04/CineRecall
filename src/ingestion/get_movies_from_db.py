"""
For ingestion, this file will handle all the operations related to db
"""

import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
import sqlite3
from sqlite3 import Connection
import logging
import pandas as pd

from src.utils import logError

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GetMoviesFromDB:

    def __init__(self):
        # load env file
        load_dotenv()
        # get the path to sqlite db
        self.sqlite_db_path = os.getenv("SQLITE_DB_PATH", "")

    def get_db_connection(self) -> Connection:
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        except Exception as e:
            logger.error(
                f"Error ocuurred while trying to get db connection for class GetMoviedFromDB: %s",
                e,
            )
            print(f"Error in getting db connection for class GetMoviesFromDB: {e}")
            logError(
                e, "GetMoviesFromDb.get_db_connection", "Error in getting db connection"
            )
            raise Exception(
                f"Acquiring db connection failed for class GetMoviedFromDB"
            ) from e

    def get_batched_details(
        self, batch_size: int, conn: Connection, last_processed_id: int = 0
    ):
        stage = "start"
        try:
            while True:
                stage = "MOVIE_FILTER"

                movie_filtered_rows = conn.execute(
                    f"""
                        SELECT movie_id
                        FROM MOVIE_PLOT
                        GROUP BY movie_id
                        HAVING
                            MAX(CASE WHEN plot_type = 'short' AND plot <> '' THEN 1 ELSE 0 END) = 1
                        AND MAX(CASE WHEN plot_type = 'detailed' AND plot <> '' THEN 1 ELSE 0 END) = 1
                        AND CAST(movie_id AS INTEGER) > CAST(? AS INTEGER)
                        ORDER BY CAST(movie_id AS INTEGER)
                        LIMIT ?
                        """,
                    (last_processed_id, batch_size),
                ).fetchall()

                # print(f"Filtered:\n", movie_filtered_rows)
                if not movie_filtered_rows:
                    logger.info("Breaking because no movie row")
                    break

                logger.info(f"We have received movie ids")
                logger.info(f"Movie ids received: {len(movie_filtered_rows)}")

                movie_ids = tuple(i[0] for i in movie_filtered_rows)
                placeholders = ",".join("?" for _ in movie_ids)

                # print("movie_ids: ", movie_ids)
                stage = "MOVIE_DETAILS"
                movie_details_rows = conn.execute(
                    f"""
                    SELECT 
                        id, tmdb_id, title, release_year, original_language, adult, tagline, book_adaptation
                    FROM
                        MOVIE_DETAILS
                    WHERE tmdb_id IN
                    (
                        {placeholders}
                    ) 
                    AND 
                    tmdb_id NOT IN
                    (	
                        SELECT tmdb_id 
                        FROM MOVIE_EMBEDDING_STATUS 
                        WHERE status = 'success'
                    )
                    ;
                    """,
                    (movie_ids),
                ).fetchall()

                if not movie_details_rows:
                    print("Breaking because no movie row")
                    break

                movie_df = pd.DataFrame(
                    movie_details_rows,
                    columns=[
                        "id",
                        "tmdb_id",
                        "title",
                        "release_year",
                        "original_language",
                        "adult",
                        "tagline",
                        "book_adaptation",
                    ],
                )

                logger.info(f"Movie details received: %s", movie_df.shape[0])

                stage = "CAST_CREW_DETAILS"
                cast_query = f"""
              
                        SELECT movie_id,
                        MAX (CASE WHEN job = 'Acting' THEN grouped END) AS actors,
                        MAX (CASE WHEN job = 'Director' THEN grouped END) AS directors,
                        MAX (CASE WHEN job = 'Producer' THEN grouped END) AS producers
                        FROM 
                        (
                        SELECT movie_id,
                        job,
                        group_concat(detail , ",") As grouped
                        FROM 
                        (  
                        SELECT 
                        M.movie_id,
                        M.crew_id,
                        M.job,
                        C.name ||
                        CASE
                            WHEN M.character IS NOT NULL AND M.character <> ''
                                THEN '|' || M.character 
                            ELSE '' 
                        END
                        ||
                        CASE
                            WHEN M.description IS NOT NULL AND M.description <> ''
                                THEN '|' || M.description
                            ELSE ''
                        END AS detail
                        FROM MOVIE_CREW_DETAILS AS M
                        JOIN CAST_CREW_DETAILS AS C
                            ON M.crew_id = C.tmdb_id
                        WHERE M.movie_id in ({placeholders})
                        AND M.job IN ('Acting', 'Director', 'Producer')
                        )
                        AS D GROUP BY D.job, D.movie_id
                        ) AS C GROUP BY movie_id
                    """
                cast_details = conn.execute(cast_query, movie_ids).fetchall()
                cast_df = pd.DataFrame(
                    [i for i in cast_details],
                    columns=["movie_id", "actors", "directors", "producers"],
                )

                logger.info("Acquired cast and crew details: %s", cast_df.shape[0])

                stage = "PLOT_DETAILS"
                plot_rows = conn.execute(
                    f"""
                                SELECT
                                    movie_id,
                                    MAX(CASE WHEN plot_type = 'short' THEN plot END) AS plot_short,
                                    MAX(CASE WHEN plot_type = 'detailed' THEN plot END) AS plot_detailed
                                FROM MOVIE_PLOT
                                WHERE movie_id IN ({placeholders})
                                GROUP BY movie_id 
                            ;
                        """,
                    (movie_ids),
                ).fetchall()
                plot_df = pd.DataFrame(
                    [i for i in plot_rows],
                    columns=["tmdb_id", "plot_short", "plot_detailed"],
                )

                logger.info("Acquired plot details: %s", plot_df.shape[0])

                stage = "GENRES"
                genre_row = conn.execute(
                    f"""
                        SELECT 
                            movie_id, 
                            group_concat(genre) 
                        FROM GENRES AS G 
                        JOIN MOVIE_GENRE_DETAILS AS MG 
                        ON G.id = MG.genre_id
                        WHERE movie_id IN ({placeholders})
                        GROUP BY movie_id
                        ;
                    """,
                    (movie_ids),
                )
                genre_df = pd.DataFrame(
                    [i for i in genre_row], columns=["tmdb_id", "genres"]
                )
                logger.info("Acquired genres: %s", genre_df.shape[0])

                stage = "MOVIE_TITLES"

                titles_row = conn.execute(
                    f"""
                                    SELECT 
                                        tmdb_id,
                                        original_title, 
                                        alt_title                                
                                    FROM MOVIE_NAME_MAPPING 
                                    WHERE tmdb_id IN ({placeholders})
                                    ;

                                """,
                    (movie_ids),
                ).fetchall()

                titles_df = pd.DataFrame(
                    [i for i in titles_row],
                    columns=["tmdb_id", "original_title", "alt_title"],
                )
                titles_df["tmdb_id"] = titles_df["tmdb_id"].apply(lambda x: str(x))

                logger.info("Acquired titles: %s", titles_df.shape[0])

                stage = "data_merge"

                complete_data = pd.merge(
                    left=pd.merge(
                        left=pd.merge(
                            left=pd.merge(
                                left=movie_df, right=plot_df, on="tmdb_id", how="inner"
                            ),
                            right=genre_df,
                            on="tmdb_id",
                            how="inner",
                        ),
                        right=titles_df,
                        on="tmdb_id",
                        how="inner",
                    ),
                    right=cast_df,
                    left_on="tmdb_id",
                    right_on="movie_id",
                    how="inner",
                )
                complete_data.drop(columns=["movie_id"], inplace=True)

                stage = "data_dict"
                data_dict = complete_data.to_dict(orient="records")

                yield data_dict
                last_processed_id = movie_ids[-1]

        except Exception as e:
            logger.error(
                f"Error ocuurred while trying to get batched data from db: %s\nError:%s",
                stage,
                e,
            )
            print(f"Error in getting batched data for stage {stage} from db: {e}")
            logError(
                e,
                "GetMoviesFromDb.get_batched_details",
                f"Error in getting batched data for stage {stage} from db",
            )


def main():
    gmd = GetMoviesFromDB()
    conn = gmd.get_db_connection()
    i = 0
    for movie in gmd.get_batched_details(20, conn):
        print(f"movie is: \n{movie}")
        print("-------")
        i += 1

        if i == 4:
            break


if __name__ == "__main__":
    main()
