import logging
import os
from datetime import datetime

# create logs directory
LOG_DIR="logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE=os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure the logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)

