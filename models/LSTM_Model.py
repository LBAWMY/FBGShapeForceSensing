import torch
import torch.nn as nn


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.fc_part1 = nn.Linear(self.hidden_size, self.output_size[0])
        self.fc_part2 = nn.Linear(self.hidden_size, self.output_size[1])
        self.fc_part3 = nn.Linear(self.hidden_size, self.output_size[2])
        self.fc_part4 = nn.Linear(self.hidden_size, self.output_size[3])
        self.fc_part5 = nn.Linear(self.hidden_size, self.output_size[4])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = x.unsqueeze(1)
        out, _ = self.lstm(x, (h0, c0))

        out1 = self.sigmoid(self.fc_part1(out[:, -1, :]))
        out2 = self.sigmoid(self.fc_part2(out[:, -1, :]))
        out3 = self.sigmoid(self.fc_part3(out[:, -1, :]))
        out4 = self.sigmoid(self.fc_part4(out[:, -1, :]))
        out5 = self.sigmoid(self.fc_part5(out[:, -1, :]))

        return out1, out2, out3, out4, out5

def test():
    # Hyperparameters
    input_size = 30
    hidden_size = 64
    num_layers = 3
    output_size = [30, 30, 1]

    # Create an instance of the LSTMNetwork
    model = LSTMNetwork(input_size, hidden_size, num_layers, output_size)

    # Print the model architecture
    print(model)

if __name__ == '__main__':
    test()