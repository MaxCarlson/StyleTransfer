import copy

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models


device = torch.device("cpu")

imageSize = 256

loader = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor()])

def loadImage(name):
    image = Image.open(name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


styleImage = loadImage('images/picasso.jpg')
contentImage = loadImage('images/hood.jpg')

print(styleImage.size())
print(contentImage.size())
assert styleImage.size() == contentImage.size()

def plotTensor(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.pause(1)

plt.figure()
plotTensor(styleImage)
plotTensor(contentImage)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()

        self.target = self.gramMatrix(target)

    def gramMatrix(self, target):
        # Calculate Gram Matrix
        a, b, c, d = target.size()
        features = input.view(a * b, c * d)
        gram = torch.matmul(features, features.t())
        gram = gram.div(a * b * c * d) # normalize matrix
        return gram

    def forward(self, input):
        gram = self.gramMatrix(input)
        self.loss = functional.mse_loss(gram, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.std    = torch.tensor(std).view(-1, 1, 1)
        self.mean   = torch.tensor(mean).view(-1, 1, 1)

    def forward(self, image):
        return (image - self.mean) / self.std

cnn = models.vgg19(True).features.to(device).eval()
normalizationMean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalizationStd = torch.tensor([0.229, 0.224, 0.225]).to(device)