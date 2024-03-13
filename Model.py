from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, BatchNorm2d, Sequential


class ConvolutionalNeuralNetwork(Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.ConvolutionalLayer_1 = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.ConvolutionalLayer_2 = Sequential(
            Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.FullyConnected = Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.ConvolutionalLayer_1(x)

        x = self.ConvolutionalLayer_2(x)

        x = x.reshape(x.size(0), -1)
        x = self.FullyConnected(x)

        return x
