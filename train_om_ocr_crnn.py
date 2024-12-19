# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 17/12/24
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from config import odometer_ocr_mode_config, odo_ocr_dataset_path, base_dataset_path
from convert_odometer_ocr_dataset import process_dataset
from metrics_utils import perform_metrics
from odometer_ocr_model import OMOcrModel, CHARSET, char_to_index, num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================= Dataset =========================
class OdometerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for group in os.listdir(root_dir):
            if "__" in group:  # Skip invalid folders
                continue
            group_path = os.path.join(root_dir, group)
            labels_path = os.path.join(group_path, "labels.txt")

            if os.path.exists(labels_path):
                with open(labels_path, "r") as file:
                    for line in file:
                        line = line.strip().split("\t")
                        if len(line) != 2:
                            print(f"Invalid line: {line} : {labels_path}")
                            continue

                        image_name, label = line
                        label = label.strip()
                        if not all(char in CHARSET for char in label):
                            print(f"Invalid label: {label} in {labels_path}")
                            continue
                        if all(c in CHARSET for c in label):
                            image_path = os.path.join(group_path, image_name)
                            if os.path.exists(image_path):
                                self.image_paths.append(image_path)
                                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        label_indices = [char_to_index[char] for char in label]
        return image, torch.tensor(label_indices, dtype=torch.long)


# ========================= Custom Collate =========================
def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels = pad_sequence(labels, batch_first=True, padding_value=num_classes - 1)
    return images, labels, label_lengths


# ========================= Main =========================
if __name__ == "__main__":
    # Config
    process_dataset(base_dataset_path, None, odo_ocr_dataset_path)
    dataset_path = odo_ocr_dataset_path
    IS_TRAINING = True
    my_model = OMOcrModel(is_training=IS_TRAINING)
    my_model.init_training_mode()
    dataset = OdometerDataset(dataset_path, my_model.dataset_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Total dataset: {len(dataset)}, tran: {train_size}, val: {val_size}")
    train_loader = DataLoader(train_dataset, batch_size=odometer_ocr_mode_config.batch, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=odometer_ocr_mode_config.batch, shuffle=False, collate_fn=custom_collate_fn)

    # Model, Loss, and Optimizer
    model = my_model.model
    # Training
    if IS_TRAINING:
        my_model.train(train_loader, val_loader, epochs=odometer_ocr_mode_config.epochs)
    # Testing
    perform_metrics(my_model, dataset, char_to_index)



"""
final_full set: Success: 2631, Failed: 369, Total testing: 3000

small set:      Success: 2412, Failed: 588, Total testing: 3000

final_fullset:   Success: 2829, Failed: 171, Total testing: 3000
Epoch 20, Train Loss: 0.1098, Val Loss: 0.1434
Total dataset images: 3846, Testing images: 3000

v2 best : Success: 2726, Failed: 274, Total testing: 3000


"""