"""
Here, we will call the LLM model to get the results.
We are going to use:
- exponential backoff with retry
- fallback strategy to use in case a model faile
- error handling
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    APIError,
    InternalServerError,
    RateLimitError,
    BadRequestError,
)
from typing import Dict, Any, Optional
import hashlib
import json
import time

load_dotenv()


class LLMClient:

    def __init__(self, cache_dir: str = "llm_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    # =========================================================================================#
    #           Operations related to storing/fetching model response to/from cache           #
    # =========================================================================================#

    def _get_cache_key(
        self, messages: list, model: str, temperature: float = 0.7
    ) -> str:
        contents = json.dumps(
            {"messages": messages, "model": model, "temperature": temperature},
            sort_keys=True,
        )

        return hashlib.md5(contents.encode()).hexdigest()

    def set(
        self, messages: list, model: str, temperature: float, result: Dict, ttl: int
    ) -> None:
        # get the cache key
        """
        Docstring for set

        :param self
        :param messages: List of input prompts that will be given to the model
        :type messages: list
        :param model: The LLM model that was used to get the response
        :type model: str
        :param temperature: Level of creativity allowed for the model
        :type temperature: float
        :param result: Data that should be stored in the cache file
        :type result: Dict
        """
        cache_file_name = self._get_cache_key(messages, model, temperature)
        cacheFile = self.cache_dir / f"{cache_file_name}.json"

        dataToStore = dict()

        """
        Cache file contents:
        "key" : <key>,
        "value":<data returned by the model>,
        "model": <model used for prediction",
        "temperature":<temperature given to the model>,
        "promptTokens":<number of tokens in the message>,
        "completedTokens": <number of tokens in the output>,
        "totalTokens":<total tokens used in this request>,
        "createdAt":<time of creation>,
        "expiresAt": <time of expiry>
        "lastAccessedAt": <time at which the data was last accessed>,
        "accessCount":<number of times this data has been accessed>
        """

        if cacheFile.exists():
            with open(cache_file_name, "r") as f:
                fileContents = json.load(f)
            dataToStore = fileContents
            dataToStore["lastAccessedAt"] = time.time()
            dataToStore["accessCount"] += 1

        else:
            dataToStore["key"] = cache_file_name
            dataToStore["value"] = result["content"]
            dataToStore["model"] = model
            dataToStore["temperature"] = temperature
            dataToStore["promptTokens"] = result["tokens"]["promptTokens"]
            dataToStore["completedTokens"] = result["tokens"]["completedTokens"]
            dataToStore["totalTokens"] = result["tokens"]["totalTokens"]
            dataToStore["createdAt"] = time.time()
            dataToStore["expiresAt"] = time.time() + ttl
            dataToStore["lastAccessedAt"] = time.time()
            dataToStore["accessCount"] = 0

        with open(cacheFile, "w") as f:
            json.dump(dataToStore, f)

    def get(
        self, messages: list, model: str, temperature: float = 0.7, ttl: int = 86400
    ) -> Any | None:
        cache_key = self._get_cache_key(messages, model, temperature)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Load cache entry
        with open(cache_file, "r") as f:
            entry = json.load(f)

        # Check expiry (your idea!)
        created_at = entry.get("created_at", 0)
        age = time.time() - created_at

        if age > ttl:
            # Expired! Delete it
            cache_file.unlink()
            return None

        return entry

    # =============================================================================================================#
    #       Operations related to fetching data from LLM model using retry and fallback with cost estimation      #
    # =============================================================================================================#

    def _call_model_with_retry(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        maxRetries: int = 3,
        initial_delay: float = 1.0,
        timeout: float = 10,
    ):

        # first, create client for the model and call it
        client = OpenAI(timeout=timeout)

        for attempt in range(maxRetries):
            try:
                model_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                )

                print(f"Got result in {attempt+1} tries")
                print("response form model: ", model_response)
                return model_response

            except RateLimitError:
                if attempt < maxRetries - 1:
                    print(
                        f"Rate limit reached for retry number: {attempt+1}/{maxRetries}"
                    )
                    print("Will try again afetr delay")
                    delay = initial_delay * (2**attempt)
                    time.sleep(delay)
                else:
                    print(f"Rate limit reached after {maxRetries} attempts")
                    raise

            except InternalServerError:
                if attempt < maxRetries - 1:
                    print(
                        f"Internal server error for retry number: {attempt+1}/{maxRetries}"
                    )
                    print("Will try again afetr delay")
                    delay = initial_delay * (2**attempt)
                    time.sleep(delay)
                else:
                    print(f"Internal server error after {maxRetries} attempts")
                    raise

            except APITimeoutError:
                if attempt < maxRetries - 1:
                    print(f"API timed out for retry number: {attempt+1}/{maxRetries}")
                    print("Will try again afetr delay")
                    delay = initial_delay * (2**attempt)
                    time.sleep(delay)
                else:
                    print(f"API timed out after {maxRetries} attempts")
                    raise

            except APIConnectionError:
                if attempt < maxRetries - 1:
                    print(
                        f"API facing connection issue for retry number: {attempt+1}/{maxRetries}"
                    )
                    print("Will try again afetr delay")
                    delay = initial_delay * (2**attempt)
                    time.sleep(delay)
                else:
                    print(f"API connection failed after {maxRetries} attempts")
                    raise

            except APIError as apie:
                print("API error: ", apie)
                if (
                    hasattr(apie, "status_code")
                    and apie.status_code  # pyright: ignore[reportAttributeAccessIssue]
                    < 500  # pyright: ignore[reportAttributeAccessIssue]
                ):
                    print(
                        f"❌ Client error (status {apie.status_code}): {apie}"  # pyright: ignore[reportAttributeAccessIssue]
                    )  # pyright: ignore[reportAttributeAccessIssue]
                    raise
                elif attempt < maxRetries - 1:
                    print(f"API Error for retry attempt {attempt+ 1}/{maxRetries}")
                    print(f"Error {str(apie)}")
                    delay = initial_delay * (2**attempt)
                    print(f"Retrying after {str(object=delay)}ms")
                    time.sleep(delay)
                else:
                    print(f"API errored out after {maxRetries} attempts")
                    raise

            except Exception as e:
                print(f"Unhandled exception occurred: {type(e)}: {e}")
                raise

        return None

    def handle_user_errors(self, error: Exception) -> str:
        error_map = {
            RateLimitError: "Our AI service is experiencing high demand. Please try again in a moment.",
            APITimeoutError: "The request took too long. Please try again with a shorter input.",
            APIConnectionError: "Unable to connect to AI service. Please check your internet connection.",
            InternalServerError: "We're having trouble processing the response. Please try again.",
            json.JSONDecodeError: "We're having trouble processing the response. Please try again.",
            "InvalidFileFormat": "Invalid file format from the URL. Please enter URL for PDF, text, or .docx files only",
        }

        # Get user-friendly message
        error_type = type(error)
        print("error type: ", error_type)
        user_message = error_map.get(
            error_type, "An unexpected error occurred. Please try again."
        )

        if user_message == "An unexpected error occurred. Please try again.":
            if str(error).__contains__("InvalidFileFormat"):
                user_message = "Invalid file format from the URL. Please enter URL for PDF, text, or .docx files only"

        # Log technical details (in production, send to logging service)
        print(f"\n[Internal Error Log]")
        print(f"Type: {error_type.__name__}")
        print(f"Message: {str(error)}")
        print(f"User sees: {user_message}\n")

        return user_message

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        completion_tokens: int,
    ) -> float:
        prices = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-4-turbo": {
                "input": 10.00 / 1_000_000,
                "output": 30.00 / 1_000_000,
            },
        }

        if model not in prices:
            model = "gpt-4o-mini"  # Default

        input_cost = input_tokens * prices[model]["input"]
        output_cost = completion_tokens * prices[model]["output"]

        return input_cost + output_cost

    def call_model_with_fallback(
        self,
        messages: list,
        primary_model: str = "gpt-4o",
        primary_max_retries: int = 1,
        primary_timeout: float = 10,
        fallback_model: str = "gpt-4o-mini",
        fallback_max_retries: int = 3,
        fallback_timeout: float = 20.0,
        temperature: float = 0.7,
        cache_ttl: int = 86400,
        use_cache: bool = True,
    ):
        try:
            print("Trying with model: ", primary_model)
            print("use cache: ", use_cache)

            resultToShow = {}
            # check cache
            if use_cache:
                # and temperature == 0.0:
                cached_result = self.get(
                    messages, primary_model, temperature, cache_ttl
                )
                if cached_result:
                    print(f"We have found ressult in the cache: {cached_result}")
                    self.set(
                        messages, primary_model, temperature, cached_result, cache_ttl
                    )

                    return cached_result["value"]

            # since we do not have cached result, let's call the API

            try:
                model_res = self._call_model_with_retry(
                    messages,
                    primary_model,
                    temperature,
                    primary_max_retries,
                    1.0,
                    primary_timeout,
                )
                print("Response from primary model: ", model_res)

                if model_res:
                    result = {
                        "content": model_res.choices[0].message.content,
                        "tokens": {
                            "promptTokens": model_res.usage.prompt_tokens,  # pyright: ignore[reportOptionalMemberAccess]
                            "completedTokens": model_res.usage.completion_tokens,  # pyright: ignore[reportOptionalMemberAccess]
                            "totalTokens": model_res.usage.total_tokens,  # pyright: ignore[reportOptionalMemberAccess]
                        },
                    }

                    if (
                        use_cache
                        and json.loads(str(model_res.choices[0].message.content)).get(
                            "confidence_score"
                        )
                        != 0.0
                    ):
                        print("Setting cache")
                        # and temperature == 0.0:
                        self.set(
                            messages, primary_model, temperature, result, cache_ttl
                        )
                    return model_res
            except (
                RateLimitError,
                APITimeoutError,
                InternalServerError,
                APIConnectionError,
            ) as e:
                print(
                    f"⚠️  Primary failed ({type(e).__name__}), trying fallback: {fallback_model}"
                )
                try:
                    fallback_res = self._call_model_with_retry(
                        messages,
                        fallback_model,
                        temperature,
                        fallback_max_retries,
                        1.0,
                        fallback_timeout,
                    )

                    print("Response from fallback model: ", fallback_res)

                    if fallback_res:
                        result = {
                            "content": fallback_res.choices[0].message.content,
                            "tokens": {
                                "promptTokens": fallback_res.usage.prompt_tokens,  # pyright: ignore[reportOptionalMemberAccess]
                                "completedTokens": fallback_res.usage.completion_tokens,  # pyright: ignore[reportOptionalMemberAccess]
                                "totalTokens": fallback_res.usage.total_tokens,  # pyright: ignore[reportOptionalMemberAccess]
                            },
                        }

                        if use_cache and temperature == 0.0:
                            self.set(
                                messages, primary_model, temperature, result, cache_ttl
                            )
                    return fallback_res
                except Exception as fallback_error:
                    raise fallback_error
            except Exception as e:
                print(f"Exception on calling with fallback: {e}")
                raise

        except Exception as e:
            messageToShow = self.handle_user_errors(e)
            print(messageToShow)


llm_client = LLMClient(cache_dir="llm_client")
