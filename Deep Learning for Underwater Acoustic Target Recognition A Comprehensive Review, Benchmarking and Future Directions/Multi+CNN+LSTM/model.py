import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN_LSTM_Model, self).__init__()

        # ---------------- Convolution Layers ----------------
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)  # -> (191)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)                     # -> (63)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2) # -> (62)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)                      # -> (20)

        # ---------------- Dropout ----------------
        self.dropout1 = nn.Dropout(0.3)

        # ---------------- LSTM Layer ----------------
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, num_layers=1, batch_first=True)

        self.dropout2 = nn.Dropout(0.3)

        # ---------------- Fully Connected ----------------
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # print("Input:", x.shape)  # (B, 1, 192)

        x = F.relu(self.conv1(x))
        # print("After Conv1:", x.shape)  # (B, 64, 191)

        x = self.pool1(x)
        # print("After Pool1:", x.shape)  # (B, 64, 63)

        x = F.relu(self.conv2(x))
        # print("After Conv2:", x.shape)  # (B, 128, 62)

        x = self.pool2(x)
        # print("After Pool2:", x.shape)  # (B, 128, 20)

        x = self.dropout1(x)
        # print("After Dropout1:", x.shape)

        x = x.permute(0, 2, 1)  # (B, 20, 128)
        # print("Permuted for LSTM:", x.shape)

        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  
        # print("After LSTM (last hidden state):", x.shape)  # (B, 32)

        x = self.dropout2(x)
        # print("After Dropout2:", x.shape)

        logits = self.fc(x)
        # print("Logits:", logits.shape)  # (B, 5)

        probs = F.softmax(logits, dim=1)
        # print("Probabilities:", probs.shape)  # (B, 5)

        return logits, probs

