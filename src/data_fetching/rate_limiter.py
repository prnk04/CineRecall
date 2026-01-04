import asyncio
import time


class AsyncRateLimiter:
    def __init__(self, rate: int, per: float = 1.0):
        """
        rate: number of requests
        per: time window in seconds
        """
        self.rate = rate
        self.per = per
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last

                # Refill tokens
                refill = (elapsed / self.per) * self.rate
                if refill > 0:
                    self._tokens = min(self.rate, self._tokens + refill)
                    self._last = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    return

                # Need to wait
                wait_time = (1 - self._tokens) * (self.per / self.rate)

            await asyncio.sleep(wait_time)
