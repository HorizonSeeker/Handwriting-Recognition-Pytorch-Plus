import torch
import torch.nn as nn

config = type('Config', (), {'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'})()

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same')
        self.bn4 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(1536, 64)  # Giữ nguyên 1536 như checkpoint

    def forward(self, images):
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        outputs = self.maxpool(self.relu(self.bn2(self.conv2(outputs))))
        outputs = self.maxpool(self.relu(self.bn3(self.conv3(outputs))))
        outputs = self.relu(self.bn4(self.conv4(outputs)))
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.reshape(
            outputs.shape[0], -1, outputs.shape[2]*outputs.shape[3])
        outputs = torch.stack([self.relu(self.fc(outputs[i]))
                               for i in range(outputs.shape[0])])
        outputs = self.dropout(outputs)
        return outputs


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, labels):
        h0 = torch.zeros(self.num_layers * 2, labels.size(0),
                         self.hidden_size).to(config.DEVICE)
        c0 = torch.zeros(self.num_layers * 2, labels.size(0),
                         self.hidden_size).to(config.DEVICE)

        outputs, _ = self.lstm(labels, (h0, c0))
        outputs = torch.stack([self.fc(outputs[i])
                               for i in range(outputs.shape[0])])
        outputs = self.softmax(outputs)
        return outputs


class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CRNN, self).__init__()
        self.RNN = BiLSTM(input_size, hidden_size,
                          output_size, num_layers)
        self.CNN = FeatureExtractor()

    def forward(self, images):
        features = self.CNN(images)
        outputs = self.RNN(features)
        return outputs