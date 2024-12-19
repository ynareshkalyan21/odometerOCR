# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 17/12/24
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.optim as optim

from config import odo_orm_model_path

CHARSET = "0123456789."
char_to_index = {char: i for i, char in enumerate(CHARSET)}
index_to_char = {i: char for char, i in char_to_index.items()}
num_classes = len(CHARSET) + 1  # Adding CTC blank index


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()
        )

        self.rnn = nn.LSTM(128 * 8, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch_size, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, w, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)


class OMOcrModel:
    def __init__(self,
                 model_path=None,
                 is_training=False):
        if not model_path:
            model_path = odo_orm_model_path
        self._init_logs(is_training)
        self.transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize according to the model's training
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CRNN(num_classes).to(self.device)
        if is_training:
            self.init_training_mode()
        else:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set the model to evaluation mode

    def _init_logs(self, is_training):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.version_dir = os.path.join("_ocr_log", f"{timestamp}{'_train' if is_training else ''}")
        os.makedirs(self.version_dir, exist_ok=True)  # Create directory if it doesn't exist
        log_file = os.path.join(self.version_dir, "log.log")
        self.log_file = open(log_file, 'a')

    def init_training_mode(self):
        self.dataset_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Random rotation and translation
            transforms.ColorJitter(brightness=0.1),  # Add brightness augmentation
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)  # Normalize to mean=0.5, std=0.5
        ])
        self.criterion = nn.CTCLoss(blank=num_classes - 1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def preprocess_image(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("L")  # Open and convert to grayscale
        else:
            image = image_path.convert("L")

        image = self.transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
        return image

    def ctc_decode(self, output_indices):
        decoded = []
        prev_idx = -1
        for idx in output_indices:
            if idx != prev_idx and idx != (num_classes - 1):  # Remove blanks and duplicates
                decoded.append(index_to_char[idx])
            prev_idx = idx
        return "".join(decoded)

    def predict(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("L")
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            image = image_path
        if image.mode != 'L':  # Check if image is not already grayscale (mode 'L')
            image = image.convert('L')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image).log_softmax(2).argmax(2)
            pred_indices = outputs.squeeze(1).cpu().numpy()
            return self.ctc_decode(pred_indices)

    def validate(self, model, val_loader, criterion, device):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels, label_lengths in val_loader:
                images, labels = images.to(device), labels.to(device)
                input_lengths = torch.full((images.size(0),), images.size(3) // 4, dtype=torch.long)

                outputs = model(images).log_softmax(2)
                loss = criterion(outputs, labels, input_lengths, label_lengths)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=10):
        model = self.model
        best_val_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for images, labels, label_lengths in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                input_lengths = torch.full((images.size(0),), images.size(3) // 4, dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = model(images).log_softmax(2)
                loss = self.criterion(outputs, labels, input_lengths, label_lengths)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            self.scheduler.step()

            # Validation
            val_loss = self.validate(model, val_loader, self.criterion, self.device)
            print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            self.log_file.write(
                f"\nEpoch {epoch + 1}, Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(model.state_dict(), os.path.join(self.version_dir, "best_model.pth"))
                print("Best model saved!")
                self.log_file.write("\n Best model saved!")
