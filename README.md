## MNIST Classification with Convolutional Neural Network (CNN) in PyTorch
This project implements a Convolutional Neural Network (CNN) for digit classification using the MNIST dataset. It covers data loading, model training, validation, testing, and saving the trained model.

## Prerequisites
Ensure the following dependencies are installed:

Python 3.x
PyTorch
torchvision
You can install the required libraries by running:
```bash
pip install torch torchvision
```

## Dataset
The dataset used is the MNIST dataset, which contains 60,000 images of handwritten digits (0-9) for training and 10,000 images for testing. The dataset is automatically downloaded by torchvision.datasets.MNIST.

If you do not already have the dataset downloaded, set download=True in the MNIST function to automatically download it.

## Code Overview
1. Load Data
The MNIST dataset is loaded using torchvision.datasets.MNIST. The training dataset is split into two sets: training (50,000 samples) and validation (10,000 samples). The test dataset is loaded separately.
Data loaders are created for training, validation, and testing with a batch size of 1000.

2. Model Creation
The ConvolutionalNeuralNetwork class defines the CNN architecture, which should be implemented in the Model.py file.

3. Loss Function and Optimizer
The Cross-Entropy loss function is used for this multi-class classification task, and the Adam optimizer is chosen with a learning rate of 0.01.

4. Training the Model
The model is trained for one epoch. During each training batch, the optimizer zeroes the gradients, computes the loss, performs backpropagation, and updates the model weights.

5. Testing the Model
The model is tested using the test dataset. The predicted labels are compared to the true labels, and the accuracy is printed.

6. Saving the Model
The trained model is saved in the Model/ directory using torch.save.

## How to Run
* Clone the repository and navigate to the project directory.
* Ensure the MNIST dataset is available in the Dataset/ directory or set download=True in the dataset loading function.
* Run the script:
```bash
python main.py
```

The script will train the CNN model, display training loss, and print test accuracy.

## Customization
* Model Architecture: You can modify the CNN architecture by editing the ConvolutionalNeuralNetwork class in Model.py.
* Hyperparameters: Adjust the batch size, learning rate, and number of epochs for different training behaviors.
* Optimizer and Scheduler: Experiment with different optimizers like SGD or include learning rate schedulers (commented out in the script).

