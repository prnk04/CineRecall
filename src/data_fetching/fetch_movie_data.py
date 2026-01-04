"""
Here, we are going to fetch the movies data from tmdb and wikipedia, and store that data in SQLite db
"""

# import necessary packages
import json
import re
import os
import time
import asyncio
import uuid
import aiohttp
import sqlite3
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

from requests import session
from datetime import datetime
from dotenv import load_dotenv
from dateutil.parser import parse
from bs4 import BeautifulSoup, Tag
import logging

from src.data_fetching.call_api import CallAPI
from src.data_fetching.store_movie_data import DatabaseOperations
from src.utils import logError

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FetchMovies:

    def __init__(
        self,
    ):
        load_dotenv()
        self.file_path_index = os.getenv("INDEX_FILE_PATH", "")
        self.file_path_titles = os.getenv("TITLES_FILE_PATH", "")
        self.merged_file_path = os.getenv("MERGED_FILE_PATH", "")
        self.format_code = "%Y-%m-%d"
        self.TMDB_API_READ_ACCESS_TOKEN = os.getenv("TMDB_API_READ_ACCESS_TOKEN")

        self.tmdb_base_url = os.getenv("TMDB_BASE_URL")
        self.wiki_base_url = os.getenv("WIKI_URL")
        self.tmdb_url_append = ["credits", "keywords"]
        self.semaphore = asyncio.Semaphore(int(os.getenv("SEMAPHORE_CONCURRENCY", 5)))

        self.movies_batch = list()
        self.crew_batch = list()
        self.roles_batch = list()
        self.plots_batch = list()
        self.genres_batch = list()
        self.movie_genre_batch = list()
        self.movie_mapping_batch = list()

    def load_file(self):
        try:

            if self.file_path_index is None or self.file_path_index == "":
                logger.error("Index movie path doesn't exixts.")
                return None

            if self.file_path_titles is None or self.file_path_titles == "":
                logger.error("Titles movie path doesn't exixts.")
                return None

            index_df = pd.read_csv(self.file_path_index)
            titles_df = pd.read_csv(self.file_path_titles)

            if index_df.shape[0] == 0 or titles_df.shape[0] == 0:
                logger.info("We do not have data")
                return None

            self.merged_df = None

            if Path(self.merged_file_path).exists():
                self.merged_df = pd.read_csv(self.merged_file_path)

            if self.merged_df is None or self.merged_df.shape[0] == 0:
                self.merged_df = pd.merge(
                    left=index_df, right=titles_df, on="movieId", how="inner"
                ).reset_index(drop=True)

                if "tmdbId" in self.merged_df:
                    self.merged_df.dropna(subset=["tmdbId"], inplace=True)
                self.merged_df.reset_index(drop=True, inplace=True)

                self.merged_df["tmdbId"] = self.merged_df["tmdbId"].astype(int)
                self.merged_df.to_csv(self.merged_file_path)
            else:
                logger.info("Merged data already exists")

            logger.info(
                "In the merged dataset, we have %s rows and %s columns",
                self.merged_df.shape[0],
                self.merged_df.shape[1],
            )
            return "exists"
        except Exception as e:
            logger.error("Error in loading files: %s", e)
            logError(e, "FetchMovies.load_files", "Error in loading files")
            raise e

    # format the data from tmdb
    def extract_tmdb_data(self, data):
        if data == None:
            data = {}
        movie_genres = data.get("genres", None)

        release_year = 0
        if data.get("release_date") is not None:
            release_year = parse(data.get("release_date", "")).year

        movie_details = {
            "imbd_id": data.get("imdb_id"),
            "tmdb_id": data.get("id"),
            # "country_of_origin": data.get("origin_country"),
            "original_language": data.get("original_language", "en"),
            "title": data.get("original_title", ""),
            "release_year": release_year,
            # datetime.strptime(data.get("release_date", ""), format_code).year,
            "tagline": data.get("tagline", ""),
            "based_on_book": "based on novel or book"
            in [x.get("name") for x in data.get("keywords", {}).get("keywords", "")],
            "adult": data.get("adult", ""),
        }

        plot_details = {"plot_brief": data.get("overview")}

        cast_crews = list()
        if data.get("credits") is not None:
            for cast in data.get("credits", {}).get("cast"):
                cast_crews.append(
                    {
                        "name": cast.get("name"),
                        "character": cast.get("character", ""),
                        "job": "Acting",
                        "description": "",
                        "id": cast.get("id", ""),
                    }
                )

            for crew in data.get("credits", {}).get("crew"):
                if crew.get("job") in [
                    "Executive Producer",
                    "Producer",
                    "Director",
                    "Musician",
                    "Screenplay",
                    "Original Story",
                ]:
                    cast_crews.append(
                        {
                            "name": crew.get("name"),
                            "job": crew.get("job", ""),
                            "character": "",
                            "description": "",
                            "id": crew.get("id", ""),
                        }
                    )

        return {
            "genres": movie_genres,
            "movies": movie_details,
            "plot_details": plot_details,
            "cast_crews": cast_crews,
        }

    def extract_wiki_data(self, wiki_data: BeautifulSoup):

        try:
            wiki_title_component = wiki_data.find("title")
            title = (
                wiki_title_component.text if wiki_title_component is not None else ""
            )

            # logger.info("Wiki: Got movie title %s", title)

            span_for_release_year = wiki_data.find_all(
                "span", class_="bday dtstart published updated itvstart"
            )

            if span_for_release_year:
                release_year = parse(span_for_release_year[0].text).year
            else:
                release_year = 0

            # logger.info("Wiki: Got movie release year %s", release_year)

            # Movie Plot
            movie_plot = ""
            plot_data = wiki_data.find_all(id="Plot")

            plot_target = plot_data[0].find_next_siblings("p") if plot_data else None
            if plot_target:
                for text in plot_target:
                    movie_plot += text.text

            plot_details = {"plot_brief": movie_plot}

            # logger.info("Wiki: Got movie plot1 %s", plot_details)

            # pattern_for_cast = re.compile(r"\b(voice\s+)?cast\b", re.IGNORECASE)

            # Movie Cast
            cast_crew_list = list()
            cast_crew_section = wiki_data.find(
                "section", attrs={"data-mw-section-id": "2"}
            )

            if cast_crew_section:
                lists = cast_crew_section.find_all("li")
                for item in lists:
                    cast_details = item.text

                    cast_crew_list.append(
                        {
                            "name": cast_details.split("as")[0].strip(),
                            "character": cast_details.split(",")[0]
                            .split("as")[1]
                            .strip(),
                            "description": (
                                ",".join(cast_details.split(",")[1:]).strip()
                                if len(cast_details.split(",")) > 1
                                else ""
                            ),
                            "job": "Acting",
                        }
                    )

            # logger.info("Wiki: Got movie cast %s", cast_crew_list)

            wiki_plot_section = wiki_data.find(
                "section", attrs={"data-mw-section-id": "1"}
            )

            wiki_plot_text = None
            if wiki_plot_section:
                wiki_plot = wiki_plot_section.find_all("p")
                wiki_plot_text = "\n\n".join([x.text for x in wiki_plot])

            if wiki_plot_text is None or wiki_plot_text == "":
                plot_details = {"plot_brief": movie_plot}
            else:
                plot_details = {"plot_brief": wiki_plot_text}
            return {
                "title": title,
                "plot_details": plot_details,
                "cast_crew_list": cast_crew_list,
                "release_year": release_year,
            }
        except Exception as e:
            print("Exception happened: ", e)
            logError(e, "FetchMovies.extract_wiki_data", "Error in careting wiki data")
            return None

    # Merge the data from tmdb and wiki
    def merge_cast_crews(self, cc_tmdb, cc_wiki):

        if cc_tmdb is None or len(cc_tmdb) == 0:
            return cc_wiki
        if cc_wiki is None or len(cc_wiki) == 0:
            return cc_tmdb
        cc_tmdb_df = pd.DataFrame(cc_tmdb)
        if "description" in cc_tmdb_df.columns.to_list():
            cc_tmdb_df.drop(columns=["description"], inplace=True)

        cc_wiki_df = pd.DataFrame(cc_wiki)
        cast_crew_merged_df = pd.merge(
            cc_tmdb_df, cc_wiki_df, on=["name", "job"], how="outer", validate="1:1"
        )
        cast_crew_merged_df["character"] = (
            cast_crew_merged_df["character_x"]
            if cast_crew_merged_df["character_x"] is not None
            else (
                cast_crew_merged_df["character_y"]
                if cast_crew_merged_df["character_y"] is not None
                else ""
            )
        )
        cast_crew_merged_df.drop(columns=["character_x", "character_y"], inplace=True)
        cast_crew_merged_df.fillna({"description": ""}, inplace=True)

        cast_crew_merged_df.dropna(subset=["id"], inplace=True)

        # print("cast: \n", cast_crew_merged_df.to_dict(orient="records"))
        return cast_crew_merged_df.to_dict(orient="records")

    def merge_results(self, original, tmdb, wiki):

        if tmdb == None:
            tmdb = {}
        if wiki == None:
            wiki = {}

        movie_title_mapping = {
            "tmdb_id": tmdb.get("movies", {}).get("tmdb_id"),
            "original_title": tmdb.get("movies", {}).get("title", ""),
            "alt_title": original.get("title", ""),
        }

        movie_data = {
            # "id": movie_id,
            "tmdb_id": (
                tmdb.get("movies", {}).get("tmdb_id")
                if tmdb.get("movies", {}).get("tmdb_id") is not None
                else original.get("tmdb_id", "")
            ),
            "imdb_id": (
                tmdb.get("movies", {}).get("imbd_id")
                if tmdb.get("movies", {}).get("imbd_id") is not None
                else original.get("imbd_id", "")
            ),
            "title": (
                tmdb.get("movies", {}).get("title", "")
                if tmdb.get("movies", {}).get("title") is not None
                else (
                    wiki.get("title")
                    if wiki.get("title") is not None
                    else original.get("title", "")
                )
            ),
            "release_year": (
                tmdb.get("movies", {}).get("release_year")
                if tmdb.get("movies", {}).get("release_year") is not None
                else wiki.get("release_year", 0)
            ),
            "original_language": (
                tmdb.get("movies", {}).get("original_language")
                if tmdb.get("movies", {}).get("original_language") is not None
                else "en"
            ),
            "adult": (
                tmdb.get("movies", {}).get("adult")
                if tmdb.get("movies", {}).get("adult") is not None
                else False
            ),
            "tagline": (
                tmdb.get("movies", {}).get("tagline")
                if tmdb.get("movies", {}).get("tagline") is not None
                else ""
            ),
            "book_adaptation": (
                tmdb.get("movies", {}).get("based_on_book")
                if tmdb.get("movies", {}).get("based_on_book") is not None
                else False
            ),
            "created_at": time.time(),
        }
        cast_and_crew = self.merge_cast_crews(
            tmdb.get("cast_crews", None), wiki.get("cast_crew_list", None)
        )

        cast_crew_details = [
            {
                # "id": x.get("uuid"),
                "name": x.get("name", ""),
                "tmdb_id": x.get("id", ""),
                "created_at": time.time(),
            }
            for x in cast_and_crew
        ]

        movie_crew_details = [
            {
                "movie_id": movie_data.get("tmdb_id"),
                "crew_id": x.get("id"),
                "character": x.get("character", ""),
                "job": x.get("job", ""),
                "decsription": x.get("description", ""),
                "created_at": time.time(),
            }
            for x in cast_and_crew
        ]

        plot_details = list()
        plot_details.append(
            {
                "plot_type": "short",
                "content": tmdb.get("plot_details", {}).get("plot_brief", ""),
            }
        )

        plot_details.append(
            {
                "plot_type": "detailed",
                "content": wiki.get("plot_details", {}).get("plot_brief", ""),
            }
        )

        movie_plot = list()
        movie_plot = [
            {
                "movie_id": movie_data.get("tmdb_id"),
                "plot_type": x.get("plot_type", ""),
                "plot": x.get("content", ""),
                "created_at": time.time(),
            }
            for x in plot_details
        ]

        genres = [
            {
                # "id": str(uuid.uuid4()),
                "name": x.get("name", ""),
                "created_at": time.time(),
            }
            for x in tmdb.get("genres", [])
        ]

        movie_genres_details = [
            {
                "movie_id": movie_data.get("tmdb_id"),
                "genre_name": x.get("name"),
                "created_at": time.time(),
            }
            for x in genres
        ]

        return {
            "movie": list(movie_data.values()),
            "cast_crew_details": [list(x.values()) for x in cast_crew_details],
            "movie_crew_details": [list(x.values()) for x in movie_crew_details],
            "genres": [list(x.values()) for x in genres],
            "movie_genres_details": [list(x.values()) for x in movie_genres_details],
            "movie_plot": [list(x.values()) for x in movie_plot],
            "movie_title_mapping": list(movie_title_mapping.values()),
        }

    async def gather_data_tmdb_wiki(self, tmdb_id, movie_name: str, session):
        try:
            tmdb_url = f"""{self.tmdb_base_url}{tmdb_id}?append_to_response={",".join(self.tmdb_url_append)}&language=en-US"""
            tmdb_headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.TMDB_API_READ_ACCESS_TOKEN}",
            }

            wiki_acceptable_movie_name = movie_name.replace(" ", "_")

            wiki_url = f"{self.wiki_base_url}{wiki_acceptable_movie_name}"
            wiki_headers = {
                "accept": "text/HTML",
                "User-Agent": "MovieRAGBot/1.0 (contact: prnkpandey00@gmail.com)",
            }

            logger.info("URL for wiki: %s", wiki_url)
            logger.info("URL for TMDB: %s", tmdb_url)

            api_instance = CallAPI(30)

            tmdb_task = api_instance.call_api(
                tmdb_url,
                session=session,
                method="GET",
                headers=tmdb_headers,
                caller="tmdb",
            )
            wiki_task = api_instance.call_api(
                wiki_url,
                session=session,
                method="GET",
                headers=wiki_headers,
                caller="wiki",
                movie_name=wiki_acceptable_movie_name,
            )

            tmdb_result, wiki_result = await asyncio.gather(
                tmdb_task, wiki_task, return_exceptions=True
            )

            tmdb_data = None
            wiki_data = None

            # print("tmdb_result:\n", tmdb_result)

            if isinstance(tmdb_result, Exception):
                print("TMDB failed", tmdb_result)
            else:
                if tmdb_result is not None:
                    tmdb_data = self.extract_tmdb_data(tmdb_result)
                else:
                    logger.info("Could not get data from TMDB")

            if isinstance(wiki_result, Exception):
                print("Wiki failed", wiki_result)
            else:
                # print("wiki res: ", wiki_result)
                if wiki_result is not None:

                    wiki_data = self.extract_wiki_data(
                        BeautifulSoup(str(wiki_result), "html.parser")
                    )
                else:
                    logger.info("Could not get data from WIKI")

            original = {"tmdb_id": tmdb_id, "title": movie_name}
            # print("tmdb data: ", tmdb_data)

            combined_movie_data = self.merge_results(original, tmdb_data, wiki_data)
            return combined_movie_data
        except Exception as e:
            print("Exception in gathering: ", e)
            logError(e, "FetchMovies.gather_data_tmdb_wiki", "Exception in gathering")

    def buffer_movies_data(self, movie_data):
        self.movies_batch.append(movie_data.get("movie"))
        self.crew_batch.extend(movie_data["cast_crew_details"])
        self.roles_batch.extend(movie_data["movie_crew_details"])
        self.plots_batch.extend(movie_data["movie_plot"])
        self.genres_batch.extend(movie_data["genres"])
        self.movie_genre_batch.extend(movie_data["movie_genres_details"])
        self.movie_mapping_batch.append(movie_data["movie_title_mapping"])

    async def start(self, max_batch_size: int = 2):

        try:
            files_loaded = self.load_file()
            if files_loaded is None or files_loaded != "exists":
                logger.info("Data doesn't exist. Stopping here")
                return None

            batched_data = list()
            failed_data = list()
            storage = list()

            dbOps = DatabaseOperations()

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with self.semaphore:
                    with dbOps.get_db_connection() as conn:
                        if self.merged_df is None:
                            logger.info(
                                "Data doesn't exist for merged file. Stopping here"
                            )
                            return None

                        for i in self.merged_df.index.to_list():
                            if i == 1000:
                                return
                            rowToUse = self.merged_df.loc[i]

                            logger.info(f"[ROW]: {rowToUse.values}")

                            this_tmdb_id = int(self.merged_df.loc[i]["tmdbId"])
                            this_movie_name = rowToUse["title"].split("(")[0].strip()

                            if this_tmdb_id and this_tmdb_id != 0:
                                this_movie_data = await self.gather_data_tmdb_wiki(
                                    this_tmdb_id, this_movie_name, session
                                )

                                if this_movie_data is not None:
                                    # batched_data.append(this_movie_data)
                                    self.buffer_movies_data(this_movie_data)

                                if len(self.movies_batch) == max_batch_size:
                                    print("The batch is full, we should store it")
                                    # some logic for storingh it
                                    try:
                                        storage.extend(batched_data)

                                        dbOps.flush_batch(
                                            conn,
                                            self.movies_batch,
                                            self.crew_batch,
                                            self.roles_batch,
                                            self.plots_batch,
                                            self.genres_batch,
                                            self.movie_genre_batch,
                                            self.movie_mapping_batch,
                                        )

                                    except Exception as e:
                                        print(
                                            f"Some exception occurred while storing data in db: {str(e)}"
                                        )
                                        logError(e, "FetchMovies.start")
                                        failed_data.extend(self.movies_batch)
                                        failed_data.extend(self.crew_batch)
                                        failed_data.extend(self.roles_batch)
                                        failed_data.extend(self.plots_batch)
                                        failed_data.extend(self.genres_batch)
                                        failed_data.extend(self.movie_genre_batch)
                                        failed_data.extend(self.movie_mapping_batch)
                                    finally:
                                        batched_data = list()
                                        print("=" * 90)
                                        self.movies_batch.clear()
                                        self.crew_batch.clear()
                                        self.roles_batch.clear()
                                        self.plots_batch.clear()
                                        self.genres_batch.clear()
                                        self.movie_genre_batch.clear()
                                        self.movie_mapping_batch.clear()

                            if len(self.movies_batch) != 0:
                                print("Seems like something is left")
                                # storage.extend(batched_data)
                                try:
                                    storage.extend(batched_data)

                                    dbOps.flush_batch(
                                        conn,
                                        self.movies_batch,
                                        self.crew_batch,
                                        self.roles_batch,
                                        self.plots_batch,
                                        self.genres_batch,
                                        self.movie_genre_batch,
                                        self.movie_mapping_batch,
                                    )

                                except Exception as e:
                                    print(
                                        f"Some exception occurred while storing data in db: {str(e)}"
                                    )
                                    logError(e, "FetchMovies.start")

                                    failed_data.append(
                                        {
                                            "movies_batch": self.movies_batch.copy(),
                                            "crew_batch": self.crew_batch.copy(),
                                            "roles_batch": self.roles_batch.copy(),
                                            "plots_batch": self.plots_batch.copy(),
                                            "genres_batch": self.genres_batch.copy(),
                                            "movie_genre_batch": self.movie_genre_batch.copy(),
                                            "movie_title_mapping": self.movie_mapping_batch.copy(),
                                        }
                                    )

                                    # failed_data.extend(self.plots_batch)
                                    # failed_data.extend(self.genres_batch)
                                    # failed_data.extend(self.movie_genre_batch)
                                finally:
                                    batched_data = list()
                                    print("=" * 90)
                                    self.movies_batch.clear()
                                    self.crew_batch.clear()
                                    self.roles_batch.clear()
                                    self.plots_batch.clear()
                                    self.genres_batch.clear()
                                    self.movie_genre_batch.clear()
                                    self.movie_mapping_batch.clear()
                        print(f"Lastly: \n{storage}")

                        # with open("testing.json", "w") as f:
                        #     f.write(str(storage))

                        with open("failed.json", "w") as f:
                            f.write(str(failed_data))
        except Exception as e:
            logger.error("Error in starting: %s", e)
            logError(e, "FetchMovies.start", "")


async def main():
    fetch_movie_instance = FetchMovies()
    await fetch_movie_instance.start()


if __name__ == "__main__":
    asyncio.run(main())
