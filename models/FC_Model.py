import torch
import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FullyConnectedNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())

        self.fc_layers = nn.Sequential(*layers)

        self.fc_part1 = nn.Linear(self.hidden_size, self.output_size[0])  # Output for part 1
        self.fc_part2 = nn.Linear(self.hidden_size, self.output_size[1])  # Output for part 2
        self.fc_part3 = nn.Linear(self.hidden_size, self.output_size[2])  # Output for part 3
        self.fc_part4 = nn.Linear(self.hidden_size, self.output_size[3])  # Output for part 4
        self.fc_part5 = nn.Linear(self.hidden_size, self.output_size[4])  # Output for part 5
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc_layers(x)
        out1 = self.sigmoid(self.fc_part1(x))
        out2 = self.sigmoid(self.fc_part2(x))
        # out3 = self.sigmoid(self.fc_part3(x))
        out3 = self.sigmoid(self.fc_part3(x))
        out4 = self.sigmoid(self.fc_part4(x))
        out5 = self.sigmoid(self.fc_part5(x))
        return out1, out2, out3, out4, out5

def test():
    # Hyperparameters
    input_size = 40
    hidden_size = 64
    num_layers = 4  # You can increase this value to have more layers
    output_size = [36, 2, 36, 1, 36]

    # Create an instance of the FullyConnectedNetwork
    model = FullyConnectedNetwork(input_size, hidden_size, num_layers, output_size)

    # Print the model architecture
    print(model)

if __name__ == '__main__':
    test()
