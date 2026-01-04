import asyncio
import time
from typing import Optional
import aiohttp
from requests import session
import requests
import logging
from bs4 import BeautifulSoup

from src.data_fetching.rate_limiter import AsyncRateLimiter
from src.utils import logError

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CallAPI:
    def __init__(self, rateLimitPerSec):
        self.rate_limiter = AsyncRateLimiter(rate=rateLimitPerSec, per=1.0)

    async def _fetch(self, session: aiohttp.ClientSession, url: str, headers: dict):
        async with session.get(url, headers=headers) as response:
            # print("response is: ", response)
            content_type = response.headers.get("Content-Type", "")

            if "text/html" in content_type:
                return response.status, await response.text()
            if "application/json" in content_type:
                return response.status, await response.json()

            return response.status, await response.read()

    def check_if_movie(self, wiki_data):
        movie_soup = BeautifulSoup(wiki_data, "html.parser")
        short_desc_div = movie_soup.find("div", attrs={"class": "shortdescription"})
        if short_desc_div is None:
            return False
        else:
            short_decs = short_desc_div.text
            if short_decs.__contains__("film"):
                return True
            else:
                return False

    async def call_api(
        self,
        url: str,
        session,
        headers: dict,
        method: str = "GET",
        body: Optional[dict] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        timeout: float = 10.0,
        message: str = "",
        caller: str = "",
        movie_name: str = "",
    ):
        for attempt in range(max_retries):
            try:
                base_url = url

                await self.rate_limiter.acquire()
                logger.info("%s Retry count: %s", caller, attempt)

                status, data = await self._fetch(session, url, headers)

                # Success
                if status < 400:
                    if caller == "wiki":
                        isMovie = self.check_if_movie(data)
                        if isMovie:
                            return data
                        else:
                            if not url.endswith("_(film)"):
                                url = base_url + "_(film)"

                    return data

                # Retryable errors
                if status in {408, 429} or status >= 500:

                    if attempt == max_retries - 1:
                        return data

                    delay = initial_delay * (2**attempt)
                    # time.sleep(delay)
                    await asyncio.sleep(delay)
                    continue

                else:
                    if caller == "wiki":
                        if attempt == max_retries - 1:
                            return data
                        if not url.endswith("_(film)"):
                            url = base_url + "_(film)"

                # Non-retryable client error
                return None

            except aiohttp.ClientError as e:
                if (
                    type(e) == requests.exceptions.HTTPError
                    and str(e).__contains__("Non-retryable error")
                ) or attempt == max_retries - 1:
                    logError(e, "CallAPI.call_api", f"Encounter Client Error")
                    return None
                time.sleep(initial_delay * (2**attempt))
            except asyncio.TimeoutError as e:

                if attempt == max_retries - 1:
                    logError(e, "CallAPI.call_api", f"Encounter Timeout Error")
                    return None
                time.sleep(initial_delay * (2**attempt))

        return None
