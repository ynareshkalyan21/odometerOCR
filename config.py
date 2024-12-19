# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 19/12/24
from dataclasses import dataclass

import yaml


def load_yaml_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


@dataclass
class ODODetectModelConfig:
    epochs: int
    imgsz: int
    batch: int
    project: str
    name: str


@dataclass
class OdoMeterOCRModeConfig:
    batch: int
    epochs: int


config_file = 'config.yaml'  # Replace with your actual config file path
config = load_yaml_config(config_file)
odo_orm_model_path = config["odo_ocr_model_path"]
odo_region_detect_model_path = config["odo_region_detect_model_path"]
base_dataset_path = config["base_dataset_path"]
test_folder = config.get("test_folder_path")
output_csv = config.get("output_csv_path"," output.csv")
odo_ocr_dataset_path = config.get("odo_ocr_dataset_path", "odo_ocr_dataset_path")

odo_detect_model_config = ODODetectModelConfig(**config['odo_detect_model'])
odometer_ocr_mode_config = OdoMeterOCRModeConfig(**config['odo_meter_ocr_mode'])
