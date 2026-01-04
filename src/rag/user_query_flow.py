"""
Here, we will implement the user query flow
    user will give us an input string
                |
    we will retrieve relevant docs
                |
            create prompt
                |
            send it to LLM
                |
        get the response
                |
        analyse the response
                |
        respond to the user
"""

import json
import os
from dotenv import load_dotenv
from pathlib import Path
import datetime

import spacy

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging

from src import get_vector_store
from src.rag.llm_client import LLMClient
from src.utils import logError

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class UserQuery:

    def __init__(self):
        load_dotenv()
        self.embedding_model = os.getenv(
            "HUGGINGFACE_EMBEDDING_MODEL", "all-mpnet-base-v2"
        )
        self.chroma_collection_name = os.getenv(
            "CHROMA_MOVIES_COLLECTION", "plotseek_movies"
        )
        self.chroma_db_version = os.getenv("CHROMA_DB_VERSION", "v0")
        self.chroma_dir = os.getenv("CHROMA_DIR", "data/chroma")
        self.vector_store = get_vector_store()

    def retrieve_movie(self, user_input, movie_expected=None):
        try:

            relevant_docs = self.vector_store.similarity_search_with_score(
                user_input, k=10
            )

            num_occurrences = 0
            total_results = 10
            lowest_position = 100
            movie_occ_dict = dict()

            logger.info("User input: %s", user_input)
            print("-" * 150)
            for i, (doc, score) in enumerate(relevant_docs):
                print(
                    f"Movie from metadata: {doc.metadata["alt_title"]}\nChunk: {doc.page_content}\nSimilarity score: {score:.3f}"
                )
                if doc.metadata["alt_title"] == movie_expected:
                    num_occurrences += 1
                    lowest_position = min(lowest_position, i)
                movie_occ_dict.update(
                    {
                        doc.metadata["alt_title"]: movie_occ_dict.get(
                            doc.metadata["alt_title"], 0
                        )
                        + 1
                    }
                )

                print("-" * 150)

            print("\n\n\n")
            print("-" * 70)
            movie_text = "Movie".center(50)
            occur_text = "Num occurrences".center(18)

            print(f"|{movie_text}|{occur_text} |")
            print("-" * 70)

            for k, v in movie_occ_dict.items():

                k_just = str(k).center(50)
                v_just = f"{str(v)}/10".center(16)
                print(f"|{k_just}|{v_just} |")
                print("-" * 70)

            print("\n\n\n")

            print(
                f"Expected movie: {movie_expected} came up {num_occurrences} number of times out of 10 results. The lowest position it hed was {lowest_position+1}"
            )

            i = 0
            max_tagline = 1
            max_short = 2
            max_title = 2
            max_detailed = 3

            plots_to_send = ""

            for doc, score in relevant_docs:
                if doc.metadata["chunkType"] == "tagline":
                    max_tagline -= 1
                    if max_tagline < 0:
                        pass
                    else:
                        plots_to_send += f"\n\n[Tagline - supporting hint]\nMovie: {doc.metadata.get('alt_title')}\n{doc.page_content}\n{"="*100}"
                elif doc.metadata["chunkType"] == "short":
                    max_short -= 1
                    if max_short < 0:
                        pass
                    else:
                        plots_to_send += f"\n\n[Short Plot- Summary]\nMovie: {doc.metadata.get('alt_title')}\n{doc.page_content}\n{"="*100}"
                if doc.metadata["chunkType"] == "detailed":
                    max_detailed -= 1
                    if max_detailed < 0:
                        pass
                    else:
                        plots_to_send += f"\n\n[Detailed Plot- Evidence]\n{doc.page_content}\n{"="*100}"
                elif doc.metadata["chunkType"] == "title":
                    max_title -= 1
                    if max_title < 0:
                        pass
                    else:
                        plots_to_send += (
                            f"\n\n[Movie Title]\n{doc.page_content}\n{"="*100}"
                        )
            print("\n\n\n")
            logger.info("Context for LLM:\n %s", plots_to_send)
            print("-" * 150)

            return

            llmClient = LLMClient()
            input_prompt = f"""
                                You are given a movie plot {user_input} by a user, who has a fuzzy memory about the movie. Given the context, figure out which movie the user is talking about.
                                Context: {plots_to_send}
                               


                                The fields [Tagline - supporting hint], [Movie Title], [Short Plot- Summary], [Detailed Plot- Evidence] can come in any order,\
                                and can be present for n number of times.
                                
                                Use [Detailed Plot- Evidence] excerpts as the primary evidence.
                                User [Short Plot- Summary] and [Movie Title] only to support or disambiguate.
                                Do not infer facts from [Movie Title] or [Tagline - supporting hint] alone.
                                Identify the movie that best matches the user's description based on the provided evidence.

                                Answer using only the provided movie excerpts. If the answer is not clearly supported, say you don't know.
                                Along with the movie title, and the year the movie was released, provide explanation as to why you inferred that the user is describing that movie. Separate the key points via newline.
                                If you are not sure about the movie year, do not provide it.
                                Also, provide the confidence score. Base the confidence score on the strength and exclusivity of the supporting detailed plot evidence compared to other provided candidates.

                                Provide the response in JSON format. Use the following as reference:
                                {{
                                    "movie_title": "Example Movie (Year)",
                                    "confidence_score": 0.0,
                                    "explanation": "Why this matches your description: 
                                    1. Key plot element A 
                                    2. Key plot element B"
                                    }}

                                The confidence_score should reflect the strength and exclusivity of the detailed plot evidence compared to other provided candidates.
                                Return ONLY valid JSON
                            """

            input_list = [
                {
                    "role": "system",
                    "content": "You are an expert in movies. You must use only the provided context. Do not infer facts that are not explicitly supported. If the answer is not clearly supported, respond with 'I don't know'. Include a confidence_score between 0.0 and 1.0 based on how strongly the provided evidence supports your conclusion. Return ONLY valid JSON.Do not wrap the JSON in markdown or code fences.",
                },
                {"role": "user", "content": input_prompt},
            ]

            res = llmClient.call_model_with_fallback(
                input_list, primary_model="gpt-4o-mini", primary_timeout=30.0
            )

            logger.info("LLM response:\n%s", res)
            if res:
                response_content = res.choices[0].message.content
                response = (
                    response_content.strip() if response_content is not None else None
                )

                return response
            else:
                return None

        except Exception as e:
            logger.error("Error in data retrieval %s", e)
            logError(e, "UserQuery.retrieve_movie", "Error in retrieving data")


def main():
    uq = UserQuery()
    uq.retrieve_movie(
        "movie where obi wan sacrifices himself",
        "Star Wars: Episode III - Revenge of the Sith",
    )


if __name__ == "__main__":
    main()
