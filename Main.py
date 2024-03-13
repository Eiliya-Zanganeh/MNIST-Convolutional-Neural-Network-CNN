from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import argmax, sum as torch_sum
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from Model import ConvolutionalNeuralNetwork
from torch import save

batch_size = 1000
lr = 0.01
epoch = 1

# Load Data

data_dataset = MNIST(
    root='./Dataset/train',
    train=True,
    transform=ToTensor(),
    download=False
)

test_dataset = MNIST(
    root='./Dataset/test',
    train=True,
    transform=ToTensor(),
    download=False
)

train_dataset, validation_dataset = random_split(data_dataset, (50000, 10000))

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Make Model

model = ConvolutionalNeuralNetwork()

# Loss Function

loss_function = CrossEntropyLoss()

# Optimizer

optimizer = Adam(model.parameters(), lr=lr)

# Lr

# lr_sch = StepLR(optimizer, 2, 0.5)

# Train Model

for num in range(epoch):
    for batch, (data_batch, label_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        out = model(data_batch)

        loss = loss_function(out, label_batch)

        loss.backward()

        optimizer.step()

        if (batch + 1) % 2 == 0:
            print(f"Epoch : {num} Batch : {batch + 1} Loss : {loss}")
            # break
    # lr_sch.step()
    # print(lr_sch.get_last_lr())


# Test Model

for data_batch, label_batch in test_loader:
    out = model(data_batch)
    predicted = argmax(out.data, 1)
    result = int(100 * torch_sum(label_batch == predicted) / batch_size)
    print(f"test {result}%")


save(model, 'Model/model.pth')
# save(model.state_dict(), 'Model/MNIST.pth')