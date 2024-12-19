

# Setup env and installation 

create env:
```bash
python3 -m venv odoOCRenv
```
activate env:
```bash
source odoOCRenv/bin/activate
```
installation
```bash
pip install ultralytics tabulate scikit-learn
```
_____________________________________
# Testing Instructions ########
##Update test_folder_path,output_csv_path the config.yaml File:
```yaml
test_folder_path: /Users/yarramsettinaresh/Downloads/train/62a4ff872be4ea4a151632b0/  #<test_folder>
output_csv_path: output.csv #output_csv
```
Run the following command
```bash
python test_predict.py

```
locate output.csv
sample output.csv
```yaml
scraped_s9V5eq_1654869235664.jpg,39846
scraped_kqZKlI_1654869276389.jpg,75263
scraped_wMzUl9_1654869167491.jpg,077440
```

_________________________________

# Training Odometer Models

## Update Dataset Path
Update the dataset path in `config.yaml`:
- Key to update: `base_dataset_path`
- New value: `/Users/yarramsettinaresh/Downloads/train/`

## Train YOLO Odometer Region Detection Model
Run the following command to train the YOLO odometer region detection model:
```bash
python train_yolo_odo_region_meter.py
```

### Script Tasks:
1. Converts the VGG dataset to the YOLO dataset format.
2. Trains a custom YOLO model to detect the odometer region.
3. Creates a new folder `yolo_model` containing training analytics.
4. Copies the trained YOLO ODO REGION model file:
   - Path: `yolo_model/odometer_lcd5/weights/best.pt`

## Build and Train the Odometer OCR Model
Run the following command:
```bash
python train_om_ocr_crnn.py
```

### Script Tasks:
1. Crops odometer images and creates a separate dataset folder `odo_ocr_dataset_path`.
2. Trains the OCR model.
3. Saves training metrics and logs in the `_ocr_log` folder.
4. Copies the trained ODOMETER OCR model file:
   - Path: `_ocr_log/<timestamp>/best_model.pth`


__________________________

## Setup Inference with Newly Trained Models
1. Copy the trained model files to the `model` folder:
   - YOLO model: `yolo_model/odometer_lcd5/weights/best.pt`
   - OCR model: `_ocr_log/<timestamp>/best_model.pth`
2. Update the `config.yaml` file with the model paths:
   - `odo_region_detect_model_path`: Path to `best.pt`
   - `odo_orm_model_path`: Path to `best_model.pth`


# configarion in config.yaml 
```yaml
######## *** 1. Mandatory Testing Config  ###################

test_folder_path: /Users/yarramsettinaresh/Downloads/train/62a4ff872be4ea4a151632b0/  #<test_folder>
output_csv_path: output.csv #output_csv

########  2.(Optional)Inference config  #######################
odo_region_detect_model_path : model/odo_region_detect_best_model.pt
odo_ocr_model_path: model/odo_ocr_best_model.pth
#######  3.(Optional) Training model optional config ########
base_dataset_path: /Users/yarramsettinaresh/Downloads/train/
odo_detect_model:
  epochs: 50
  imgsz: 640
  batch: 16
  project: yolo_model # logs parent folder name
  name: odometer_lcd  # logs folder name

odo_meter_ocr_mode:
  batch: 16
  epochs: 20
```
