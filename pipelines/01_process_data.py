import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.data.processor import DataProcessor

def run_processing():
    """Loads configuration, initializes DataProcessor, and runs processing."""
    print("--- Starting Data Processing ---")
    config = load_config("config/config.yml")
    
    # First, combine raw CSVs into one.
    # This logic is simple enough to be here, but could be moved to processor if it gets complex.
    combine_raw_csvs(config)

    processor = DataProcessor(config)
    processor.process()
    
    print("--- Data Processing Finished ---")

def combine_raw_csvs(config):
    import pandas as pd
    import glob

    print("Combining raw data files...")
    raw_data_dir = config["data"]["raw_data_dir"]
    glob_pattern = config["data"]["raw_glob_pattern"]
    output_path = config["data"]["combined_raw_path"]

    file_paths = glob.glob(f"{raw_data_dir}/{glob_pattern}")
    if not file_paths:
        print(f"No files found for pattern: {glob_pattern} in {raw_data_dir}")
        return

    df_list = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined data saved to {output_path}")


if __name__ == "__main__":
    run_processing()
