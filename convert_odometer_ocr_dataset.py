# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 19/12/24
# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 14/12/24
import copy
import os
import json
from PIL import Image

from config import base_dataset_path, odo_ocr_dataset_path


def adjust_polygon_coords(coords, crop_x_min, crop_y_min):
    """Adjust polygon coordinates based on cropping."""
    adjusted_coords = {
        "all_points_x": [x - crop_x_min for x in coords["all_points_x"]],
        "all_points_y": [y - crop_y_min for y in coords["all_points_y"]]
    }
    return adjusted_coords


def process_group(group_path, output_path, text_output_path):
    """Process a single group of images."""
    region_file = os.path.join(group_path, "via_region_data.json")

    # Load the via_region_data.json file
    with open(region_file, "r") as f:
        regions_data = json.load(f)

    updated_regions = {}
    text_labels = []

    for img_name, img_data in regions_data.items():
        original_img_data = img_data
        img_data = copy.deepcopy(img_data)
        img_path = os.path.join(group_path, img_data["filename"].split("/")[-1])
        img = Image.open(img_path)
        odo_coords_region = None

        for region in img_data["regions"]:
            if region["region_attributes"]["identity"] == "LCD":
                # Process LCD Region
                lcd_coords = region["shape_attributes"]
                crop_x_min = min(lcd_coords["all_points_x"])
                crop_x_max = max(lcd_coords["all_points_x"])
                crop_y_min = min(lcd_coords["all_points_y"])
                crop_y_max = max(lcd_coords["all_points_y"])

                # Crop LCD region
                cropped_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

                # Save cropped LCD image
                if output_path:
                    cropped_img_dir = os.path.join(output_path, os.path.basename(group_path))
                    os.makedirs(cropped_img_dir, exist_ok=True)
                    cropped_img_path = os.path.join(cropped_img_dir, img_data["filename"])
                    cropped_img.save(cropped_img_path)


                # Add updated data to new JSON
                updated_regions[img_data["filename"]] = img_data

            if region["region_attributes"]["identity"] == "odometer":
                # Process Odometer Region
                odo_coords_region = region
        if odo_coords_region:
            odo_coords = odo_coords_region["shape_attributes"]
            crop_x_min = min(odo_coords["all_points_x"])
            crop_x_max = max(odo_coords["all_points_x"])
            crop_y_min = min(odo_coords["all_points_y"])
            crop_y_max = max(odo_coords["all_points_y"])
            # Save cropped odometer image
            if text_output_path:
                image_name = img_data["filename"].split("/")[-1]
                cropped_odometer = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                text_img_dir = os.path.join(text_output_path, os.path.basename(group_path))
                os.makedirs(text_img_dir, exist_ok=True)
                cropped_odo_path = os.path.join(text_img_dir, image_name)
                cropped_odometer.save(cropped_odo_path)

                adjusted_coords = adjust_polygon_coords(
                    odo_coords, crop_x_min, crop_y_min
                )
                odo_coords_region["shape_attributes"]["all_points_x"] = adjusted_coords["all_points_x"]
                odo_coords_region["shape_attributes"]["all_points_y"] = adjusted_coords["all_points_y"]

                # Extract the reading from the region attributes
                reading = odo_coords_region["region_attributes"].get("reading", "unknown")
                text_labels.append(f"{image_name}\t{reading}")
                if not reading:
                    print(f"empty odometer reding  {img_path}")
        else:
            print(f"No odo_coords_region {img_path}")

    # Save the updated via_region_data.json for LCD dataset
    if output_path:
        output_region_file = os.path.join(output_path, os.path.basename(group_path), "via_region_data.json")
        os.makedirs(os.path.dirname(output_region_file), exist_ok=True)
        with open(output_region_file, "w") as f:
            json.dump(updated_regions, f, indent=4)
    if text_output_path:
        # Save the text labels for odometer text detection dataset
        text_labels_file = os.path.join(text_output_path, os.path.basename(group_path), "labels.txt")
        os.makedirs(os.path.dirname(text_labels_file), exist_ok=True)
        with open(text_labels_file, "w") as f:
            f.write("\n".join(text_labels))


def process_dataset(dataset_path, output_path, text_output_path):
    """Process the entire dataset."""
    groups = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for group in groups:
        group_path = os.path.join(dataset_path, group)
        process_group(group_path, output_path, text_output_path)


if __name__ == "__main__":
    # Process the dataset
    process_dataset(base_dataset_path, None, odo_ocr_dataset_path)
