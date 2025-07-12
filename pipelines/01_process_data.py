import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.data.processor import DataProcessor

def run_processing():
    """Loads configuration and runs the data processing pipeline."""
    config = load_config("config/config.yml")
    processor = DataProcessor(config)
    processor.process_raw_data()

if __name__ == "__main__":
    run_processing()