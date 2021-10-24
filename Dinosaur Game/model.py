import torch


class BasicNNet(torch.nn.Module):
    """ Basic NNet for playing the dino game """

    def __init__(self, input_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, 500)
        self.activation1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(500, 250)
        self.activation2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(250, 25)
        self.activation3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Linear(25, 3)

    def forward(self, x):
        """ Forward pass on images to calculate log-probability of each key press given image pixels"""
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.layer4(x)

        log_probs = torch.nn.functional.log_softmax(x, dim=1)

        return log_probs
