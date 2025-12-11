import logging
import os
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the directory for log files
log_dir = "../Logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Update log_filename to include the directory path
log_filename = os.path.join(log_dir, f"baseline_evaluation_{timestamp}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__) 