import yaml
from typing import Dict


def load_config(config_path: str) -> Dict:
    """YAML 설정 파일을 읽어 딕셔너리로 반환합니다."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None
