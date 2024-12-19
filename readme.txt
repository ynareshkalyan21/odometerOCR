#############    Environment Setup    ##################

1.Create a Virtual Environment:
python3 -m venv odoOCRenv

2.Activate the Environment:
source odoOCRenv/bin/activate

3.Install Dependencies:
pip install ultralytics tabulate sklearn

###############  Testing Instructions ########
    --> Update the config.yaml File:
        --> update "test_folder_path" value  in config.yaml
        --> update "output_csv_path" value  in config.yaml
    --> Run test_predict.py : python test_predict.py

    Example:
        test_folder_path: /Users/yarramsettinaresh/Downloads/train/62a4ff872be4ea4a151632b0/  #<test_folder>
        output_csv_path: output.csv  #<output_csv>


################### Training #############
    Step1: Update the dataset path in config.yaml
        change the training dataset path configuration in confg.yaml  key :
        base_dataset_path = "/Users/yarramsettinaresh/Downloads/train/"

    Step2: Train YOLO Odometer Region Detection Model
        --> 1)  Run the following command : python train_yolo_odo_region_meter.py
                Script Tasks:
                  -->  a. Converts the VGG dataset to the YOLO dataset format: output folder is "_yolo_odo_region_dataset"
                  -->  b. Trains a custom YOLO model with dataset(_yolo_odo_region_dataset) to detect the odometer region.
        --> 2)  Copies the trained YOLO ODO REGION model file to model folder: Path: yolo_model/odometer_lcd5/weights/best.pt



    Step3: Build Training ODOMeter OCR model:
        --> 1) Run the following command : python train_om_ocr_crnn.py
            Script Tasks:
                --> a. Crops odometer images and creates a separate dataset folder odo_ocr_dataset_path : output folder is "odo_ocr_dataset_path"
                --> b. Trains the OCR mode
                --> c. Saves training metrics and logs in the "_ocr_log" folder
        --> 2) Copies the trained ODOMETER OCR model file: "_ocr_log/<timestamp>/best_model.pth" to "model/odo_orm_best_model.pth" folder


################### *** Setup Inference with Newly Trained Models *** ###################
  -->1) Copy the trained model files to the model folder
        --> a. YOLO model: yolo_model/odometer_lcd5/weights/best.pt
        --> b. OCR model: _ocr_log/<timestamp>/best_model.pth
  -->2) Update the config.yaml file with the model paths:
        --> a. odo_region_detect_model_path: Path to best.pt
        --> b. odo_orm_model_path: Path to best_model.pth





