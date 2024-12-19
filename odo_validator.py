import csv
import json
import os
import ssl
import certifi
from collections import defaultdict, Counter
from PIL import Image
import cv2
import numpy as np
from tabulate import tabulate


# Initialize OCR Model
from config import base_dataset_path
from model_inference import DetectOMRegionAndOCR
from odometer_ocr_model import OMOcrModel

oomm = OMOcrModel()
ssl_context = ssl.create_default_context(cafile=certifi.where())
yolo_omr_model = DetectOMRegionAndOCR()


def save_error_details_to_csv(error_details, output_csv_path):
    """
    Save error details to a CSV file.

    Args:
        error_details (list of dict): List of error details with keys:
            'group_name', 'image_url', 'predicted_value', 'expected_value'
        output_csv_path (str): Path to the output CSV file.
    """
    # Define CSV headers
    # headers = ['Group Name', 'Image URL', 'Predicted Value', 'Expected Value', "yolo_value"]
    headers = list(error_details[0].keys())
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    # Write to the CSV file
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(error_details)


def is_odo_inside_lcd(odo_coords, lcd_coords):
    crop_x_min, crop_x_max, crop_y_min, crop_y_max = odo_coords
    lcd_crop_x_min, lcd_crop_x_max, lcd_crop_y_min, lcd_crop_y_max = lcd_coords
    return (crop_x_min >= lcd_crop_x_min and crop_x_max <= lcd_crop_x_max and
            crop_y_min >= lcd_crop_y_min and crop_y_max <= lcd_crop_y_max)


import csv

def print_group_metrics_table(group_metrics, metrics_csv_path):
    """
    Prints a table summarizing the metrics for each group and saves it to a CSV file.

    Args:
        group_metrics: A dictionary where keys are group names and values are dictionaries
                       containing the following metrics:
                           - 'total': Total number of samples in the group.
                           - 'my_model_count': Number of samples correctly labeled by the model.
                           - 'yolo_count': Number of samples correctly labeled by YOLO.
                           - 'yolo_omr_model_error': Number of samples with no YOLO prediction.
                           - 'data_error': Number of samples with label discrepancies between label OCR and YOLO detected OCR.
                           - 'error': Number of samples with empty labels.
        metrics_csv_path: Path to the CSV file where the metrics will be saved.

    Returns:
        None
    """

    table_data = []
    total_samples = 0
    total_my_model_correct = 0
    total_yolo_correct = 0
    total_no_yolo_prediction = 0
    total_label_discrepancies = 0
    total_empty_labels = 0

    for group_name, metrics in group_metrics.items():
        total_samples += metrics['total']
        total_my_model_correct += metrics['my_model_count']
        total_yolo_correct += metrics['yolo_count']
        total_no_yolo_prediction += metrics['yolo_omr_model_error']
        total_label_discrepancies += metrics['data_error']
        total_empty_labels += metrics['error']

        yolo_percent = (metrics['yolo_count'] / metrics['total']) * 100 if metrics['total'] != 0 else 0
        my_model__per = (metrics['my_model_count'] / metrics['total']) * 100 if metrics['total'] != 0 else 0
        mismatch_per = (metrics['data_error'] / metrics['total']) * 100 if metrics['total'] != 0 else 0

        table_data.append([
            group_name,
            metrics['total'],
            f"{metrics['my_model_count']} ({my_model__per:.2f}%)",
            f"{metrics['yolo_count']} ({yolo_percent:.2f}%)",
            metrics['yolo_omr_model_error'],
            f"{metrics['data_error']} ({mismatch_per:.2f}%)",
            metrics['error'],
        ])

    # Calculate overall percentages
    overall_my_model_accuracy = (total_my_model_correct / total_samples) * 100 if total_samples != 0 else 0
    overall_yolo_accuracy = (total_yolo_correct / total_samples) * 100 if total_samples != 0 else 0

    # Add overall summary to the table
    table_data.append([
        "**Total**",
        total_samples,
        f"{total_my_model_correct} ({overall_my_model_accuracy:.2f}%)",
        f"{total_yolo_correct} ({overall_yolo_accuracy:.2f}%)",
        total_no_yolo_prediction,
        f"{total_label_discrepancies} ({(total_label_discrepancies / total_samples) * 100:.2f}%)",
        total_empty_labels,
    ])

    # Define table headers
    headers = [
        "Group",
        "Tot",
        "label OCR",
        "YOLO OCR",
        "empty YOLO-OCR",
        "missmatch(L-OCR vs YOLO-OCR)",
        "No label",
    ]

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Save metrics to CSV file
    # os.makedirs(metrics_csv_path, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    with open(metrics_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)
        csv_writer.writerows(table_data)

    print(f"Metrics saved to {metrics_csv_path}")

def parse_json_files_recursive(root_folder):
    error_details = []
    # Initialize group-wise metrics
    group_metrics = defaultdict(lambda: {
        'my_model_count': 0,
        'yolo_count': 0,
        'data_error': 0,
        'pytesseract_count': 0,
        'error': 0,
        'pytesseract_over_success': 0,
        'yolo_omr_model_error': 0,
        'total': 0,
    })

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                parent_path = file_path.replace(os.path.basename(file_path), "")
                group_name = parent_path.split("/")[-2]  # Use parent folder name as group
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for image_data in data.values():
                        odometer = None
                        LCD = None
                        odometer_dd = None

                        for region in image_data['regions']:
                            identity = region['region_attributes']['identity']
                            if identity == 'odometer':
                                reading = region['region_attributes'].get('reading')
                                odometer_dd = reading
                                odometer = region
                            elif identity == 'LCD':
                                LCD = region

                        if LCD and odometer:
                            odo_coords = odometer["shape_attributes"]
                            crop_x_min = min(odo_coords["all_points_x"])
                            crop_x_max = max(odo_coords["all_points_x"])
                            crop_y_min = min(odo_coords["all_points_y"])
                            crop_y_max = max(odo_coords["all_points_y"])

                            lcd_coords = LCD["shape_attributes"]
                            lcd_crop_x_min = min(lcd_coords["all_points_x"])
                            lcd_crop_x_max = max(lcd_coords["all_points_x"])
                            lcd_crop_y_min = min(lcd_coords["all_points_y"])
                            lcd_crop_y_max = max(lcd_coords["all_points_y"])

                            odo_bounds = (crop_x_min, crop_x_max, crop_y_min, crop_y_max)
                            lcd_bounds = (lcd_crop_x_min, lcd_crop_x_max, lcd_crop_y_min, lcd_crop_y_max)

                            is_inside = is_odo_inside_lcd(odo_bounds, lcd_bounds)
                            if not is_inside:
                                print(f"Odo is outside: {image_data['filename']}")

                            img_path = os.path.join(parent_path, image_data['filename'].split("/")[-1])
                            img = Image.open(img_path)
                            cropped_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                            gray_image = cropped_img.convert('L')
                            gray_image_array = np.array(gray_image)
                            thresh_image = cv2.adaptiveThreshold(
                                gray_image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                            )
                            scale_percent = 200
                            width = int(thresh_image.shape[1] * scale_percent / 100)
                            height = int(thresh_image.shape[0] * scale_percent / 100)
                            odometer_reading = oomm.predict(cropped_img)
                            yolo_detect = yolo_omr_model.predict(img)
                            # yolo_detect = odometer_reading
                            group = group_metrics[group_name]
                            group["total"] += 1
                            if not yolo_detect:
                                group["yolo_omr_model_error"] += 1
                            if yolo_detect == odometer_dd:
                                group['yolo_count'] += 1
                            if not yolo_detect == odometer_reading:
                                group['data_error'] += 1
                            if odometer_reading == odometer_dd:
                                group['my_model_count'] += 1

                            elif odometer_dd:
                                group['error'] += 1
                            if not odometer_dd == odometer_reading:
                                error_details.append({
                                    'group_name': group_name,
                                    'image_url': img_path,
                                    'predicted_value': odometer_reading,
                                    'expected_value': odometer_dd,
                                    'yolo_value': yolo_detect
                                })

    return group_metrics, error_details


def print_group_metrics(group_metrics):
    for group_name, metrics in group_metrics.items():
        print(f"\n********* Group: {group_name} *********")
        print(f"My Model Correct Predictions: {metrics['my_model_count']}")
        print(f"YOLO Correct Predictions: {metrics['yolo_count']}")
        print(f"Data Errors: {metrics['data_error']}")
        print(f"Pytesseract Correct Predictions: {metrics['pytesseract_count']}")
        print(f"Errors: {metrics['error']}")
        print(f"Pytesseract Additional Success: {metrics['pytesseract_over_success']}\n")


if __name__ == "__main__":
    root_folder =  base_dataset_path # Replace with your root folder
    root_folder = os.path.join(root_folder, "62a4ff852be4ea4a151632a7/")
    print("Starting dataset analysis...")

    output_csv_path = f"log/{root_folder.split('/')[-2]}/error.csv"  # Define the output CSV file path
    metrics_csv_path = f"log/{root_folder.split('/')[-2]}/metrics.csv"

    print("Starting dataset analysis...")
    group_metrics, error_details = parse_json_files_recursive(root_folder)

    print("\n=== Group-wise Metrics ===")
    print_group_metrics_table(group_metrics, metrics_csv_path)

    print("\n=== Saving Error Details to CSV ===")
    save_error_details_to_csv(error_details, output_csv_path)
    print(f"details saved to: {output_csv_path}")
