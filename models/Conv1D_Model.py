import torch
import torch.nn as nn


class Conv1DNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(Conv1DNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU()
        )

        # Fully connected layers for each output part
        self.fc_part1 = nn.Linear(1536, self.output_size[0])
        self.fc_part2 = nn.Linear(1536, self.output_size[1])
        self.fc_part3 = nn.Linear(1536, self.output_size[2])
        self.fc_part4 = nn.Linear(1536, self.output_size[3])
        self.fc_part5 = nn.Linear(1536, self.output_size[4])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        out1 = self.sigmoid(self.fc_part1(x))
        out2 = self.sigmoid(self.fc_part2(x))
        out3 = self.sigmoid(self.fc_part3(x))
        out4 = self.sigmoid(self.fc_part4(x))
        out5 = self.sigmoid(self.fc_part5(x))
        # out1 = self.fc_part1(x)
        # out2 = self.fc_part2(x)
        # out3 = self.fc_part3(x)
        # out4 = self.fc_part4(x)
        # out5 = self.fc_part5(x)
        return out1, out2, out3, out4, out5

def test():
    # Hyperparameters
    input_size = 30
    output_size = [30, 30, 1]
    # Create an instance of the Conv1DNetwork
    model = Conv1DNetwork(input_size, output_size)

    # Print the model architecture
    print(model)

if __name__ == '__main__':
    test()