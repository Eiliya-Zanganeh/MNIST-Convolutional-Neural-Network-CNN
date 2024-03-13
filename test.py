from PIL import Image
import torch
import torchvision.transforms as transforms

# from Model import ConvolutionalNeuralNetwork
# model = ConvolutionalNeuralNetwork()
# model.load_state_dict(torch.load('model_weights.pth'))


model = torch.load('Model/model.pth')

image_path = 'img.png'
image = Image.open(image_path)

preprocess = transforms.Compose([
    transforms.Resize(28),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)

predicted_class = output.argmax(dim=1)
print(output)
print(predicted_class)
