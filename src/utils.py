import os
from pathlib import Path
import datetime


def logError(errorMessage: Exception, func_name: str, additionalMsg: str = ""):
    try:
        log_dir = Path(f"logs/{datetime.date.today().strftime('%Y_%m_%d')}")
        log_dir.mkdir(parents=True, exist_ok=True)

        message_To_store = f"\n{datetime.date.today().strftime("%Y_%m_%d_%H:%M:%S")} {str(additionalMsg)}: {str(func_name)}: {str(errorMessage)}"

        with open(
            log_dir
            / f"error_logs_{datetime.datetime.now().strftime('%Y_%m_%d_%H')}.txt",
            "a",
        ) as f:
            f.write(message_To_store)
    except Exception as e:
        print("Error in logging error: ", e)
