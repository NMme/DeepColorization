import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepColorSimple(nn.Module):
    def __init__(self):
        super(DeepColorSimple, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        #pool
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        #pool
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        #pool
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        #pool
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(32*2*2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1024)

        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=(5, 5), padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 32 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 1, 32, 32)
        x = torch.sigmoid(self.deconv1(x))
        return x