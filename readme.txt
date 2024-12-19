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
        --> a. odo_region_detect_model model: yolo_model/odometer_lcd5/weights/best.pt
        --> b.  odo_orm_model: _ocr_log/<timestamp>/best_model.pth
  -->2) Update the config.yaml file with the model paths:
        --> a. odo_region_detect_model_path: Path to best.pt
        --> b. odo_ocr_model_path: Path to best_model.pth

######### Training Metrics insights  ###########
   ---> "python train_om_ocr_crnn.py" will provide F1 score details
   ---> sample output for given dataset (https://drive.google.com/file/d/12dXSLNwbArubn25XjcMBd-Pqw7go4ShN/vie
w?usp=sharing)
           Character-wise Metrics sorted by F1 Score:
        Character      F1 Score    Precision    Recall
        -----------  ----------  -----------  --------
        4                  0.98         0.99      0.98
        2                  0.98         0.98      0.98
        3                  0.98         0.98      0.99
        7                  0.98         0.98      0.98
        5                  0.98         0.99      0.97
        1                  0.98         0.97      0.98
        6                  0.97         0.98      0.97
        0                  0.97         0.97      0.97
        8                  0.97         0.97      0.97
        9                  0.97         0.97      0.97
        .                  0.81         0.95      0.70

        Success: 3593, Failed: 253, Total testing: 3846
        Accuracy: 93.42%

########  validation metrics #########
      ---> "python train_om_ocr_crnn.py" will accuracy details
      ---> sample output for given dataset (https://drive.google.com/file/d/12dXSLNwbArubn25XjcMBd-Pqw7go4ShN/vie
w?usp=sharing)
            === Group-wise Metrics ===
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | Group                    |   Tot | label OCR     | YOLO OCR      |   empty YOLO-OCR | missmatch(L-OCR vs YOLO-OCR)   |   No label |
            +==========================+=======+===============+===============+==================+================================+============+
            | 62a4ff872be4ea4a151632b0 |   193 | 176 (91.19%)  | 172 (89.12%)  |                3 | 21 (10.88%)                    |         17 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff872be4ea4a151632af |   193 | 176 (91.19%)  | 172 (89.12%)  |                3 | 17 (8.81%)                     |         17 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff862be4ea4a151632aa |   191 | 171 (89.53%)  | 162 (84.82%)  |                4 | 20 (10.47%)                    |         20 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff862be4ea4a151632a9 |   193 | 184 (95.34%)  | 171 (88.60%)  |                5 | 19 (9.84%)                     |          9 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a501c72be4ea4a151632bb |   191 | 164 (85.86%)  | 158 (82.72%)  |                5 | 18 (9.42%)                     |         27 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a501c72be4ea4a151632bc |   192 | 178 (92.71%)  | 175 (91.15%)  |                6 | 20 (10.42%)                    |         14 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a501c62be4ea4a151632ba |   191 | 173 (90.58%)  | 161 (84.29%)  |                2 | 23 (12.04%)                    |         18 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a501c62be4ea4a151632b8 |   194 | 163 (84.02%)  | 162 (83.51%)  |                6 | 33 (17.01%)                    |         27 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a501c62be4ea4a151632b6 |   187 | 175 (93.58%)  | 176 (94.12%)  |                3 | 17 (9.09%)                     |         12 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a501c62be4ea4a151632b7 |   187 | 164 (87.70%)  | 170 (90.91%)  |                3 | 30 (16.04%)                    |         23 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a501c62be4ea4a151632b9 |   198 | 189 (95.45%)  | 185 (93.43%)  |                1 | 17 (8.59%)                     |          9 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff872be4ea4a151632b4 |   188 | 180 (95.74%)  | 176 (93.62%)  |                2 | 15 (7.98%)                     |          8 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff872be4ea4a151632b3 |   195 | 174 (89.23%)  | 165 (84.62%)  |                4 | 26 (13.33%)                    |         20 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff872be4ea4a151632b2 |   194 | 185 (95.36%)  | 188 (96.91%)  |                1 | 13 (6.70%)                     |          9 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff872be4ea4a151632b5 |   194 | 182 (93.81%)  | 173 (89.18%)  |                3 | 19 (9.79%)                     |         12 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff862be4ea4a151632ab |   194 | 181 (93.30%)  | 176 (90.72%)  |                4 | 14 (7.22%)                     |         13 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff862be4ea4a151632ad |   194 | 181 (93.30%)  | 167 (86.08%)  |                3 | 27 (13.92%)                    |         13 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff862be4ea4a151632ac |   195 | 185 (94.87%)  | 178 (91.28%)  |                3 | 13 (6.67%)                     |         10 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff852be4ea4a151632a8 |   193 | 160 (82.90%)  | 168 (87.05%)  |                1 | 38 (19.69%)                    |         33 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | 62a4ff852be4ea4a151632a7 |   194 | 181 (93.30%)  | 180 (92.78%)  |                2 | 15 (7.73%)                     |         13 |
            +--------------------------+-------+---------------+---------------+------------------+--------------------------------+------------+
            | **Total**                |  3851 | 3522 (91.46%) | 3435 (89.20%) |               64 | 415 (10.78%)                   |        324 |







