# Created by Yarramsettinaresh, Goraka Digital Private Limited, 19/12/24
import os
import logging
import pandas as pd

# Configure logging
from config import test_folder, output_csv
from model_inference import DetectOMRegionAndOCR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Path to test dataset and output CSV
# Initialize the odometer detection and OCR model
pipeline = DetectOMRegionAndOCR()


def read_odometer(image_path):
    """
    Reads the odometer reading from the image.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: Predicted odometer reading or an error message.
    """
    try:
        logging.info(f"Processing image: {image_path}")
        odometer_text = pipeline.predict(image_path)
        return odometer_text.strip()
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return "Error"


def predict(folder_path, predictions):
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isdir(image_path):
            continue
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            predicted_reading = read_odometer(image_path)
            predictions.append({'filename': image_name, 'predicted_reading': predicted_reading})
        else:
            logging.warning(f"Skipping unsupported file format: {image_name}")


def main():
    """
    Main function to process images, predict odometer readings,
    and save results to a CSV file.
    """
    if not os.path.exists(test_folder):
        logging.error(f"Test folder '{test_folder}' does not exist.")
        return

    predictions = []
    predict(test_folder, predictions)
    for folder_name in os.listdir(test_folder):
        folder_path = os.path.join(test_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue
        predict(folder_path, predictions)
    # Save predictions to a CSV file
    if predictions:
        df = pd.DataFrame(predictions)
        df.to_csv(output_csv, index=True, header=False)
        logging.info(f"Predictions saved to {output_csv}")
    else:
        logging.warning("No valid predictions generated. CSV file was not created.")


if __name__ == "__main__":
    main()
