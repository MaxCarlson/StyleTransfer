import copy

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models


device = torch.device("cuda")

imageSize = 512

loader = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor()])

def loadImages(image1Name, image2Name):
    image1 = Image.open(image1Name)
    image2 = Image.open(image2Name)

    h1, v1 = image1.size
    h2, v2 = image2.size

    width = h1
    height = v1
    ideal_width = h2
    ideal_height = v2
    ideal_aspect = h2 / float(v2)
    aspect = h1 / float(v1)
    if aspect > ideal_aspect:
        # Then crop the left and right edges:
        new_width = int(ideal_aspect * height)
        offset = (width - new_width) / 2
        resize = (offset, 0, width - offset, height)
    else:
        # ... crop the top and bottom:
        new_height = int(width / ideal_aspect)
        offset = (height - new_height) / 2
        resize = (0, offset, width, height - offset)

    image1 = image1.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)

    image1 = loader(image1).unsqueeze(0)
    image2 = loader(image2).unsqueeze(0)

    return image1.to(device, torch.float), image2.to(device, torch.float)


styleImage, contentImage = loadImages('images/picasso.jpg', 'images/me3.jpg')


print(styleImage.size())
print(contentImage.size())
assert styleImage.size() == contentImage.size()

def plotTensor(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)

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
        features = target.view(a * b, c * d)
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

cnn         = models.vgg19(True).features.to(device).eval()
normMean    = torch.tensor([0.485, 0.456, 0.406]).to(device)
normStd     = torch.tensor([0.229, 0.224, 0.225]).to(device)

contentLayers   = ['conv4']
styleLayers     = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

def getLayer(layer, i):
    layerName = ''
    if isinstance(layer, nn.Conv2d):
        i += 1
        layerName = 'conv{}'.format(i)

    elif isinstance(layer, nn.ReLU):
        layerName   = 'relu{}'.format(i)
        layer       = nn.ReLU(inplace=False)

    elif isinstance(layer, nn.MaxPool2d):

        # Can set pooling type here. Default is Max2d
        layerName = 'maxpool{}'.format(i)

        #layerName = 'avgpool{}'.format(i)
        layer = nn.AvgPool2d(layer.kernel_size)
    
    elif isinstance(layer, nn.BatchNorm2d):
        layerName = 'bachnorm{}'.format(i)

    return layer, layerName, i

def addLayer(i, image, model, LossType, lossName, losses):
    target  = model(image).detach()
    loss    = LossType(target)
    model.add_module(lossName + str(i), loss)
    losses.append(loss)

def createModel(cnn, normMean, normStd, styleImage, contentImage):

    normalizer = Normalization(normMean, normStd).to(device)

    contentLosses = []
    styleLosses = []

    # Start building out model (normalize inputs first  with the Normilzation layer)
    model = nn.Sequential(normalizer)

    # Walk through the vgg model layer by layer
    i = 0
    ll = 0
    for idx, layer in enumerate(cnn.children()):

        # Here we get the current layer, and can interact 
        # with/change the layer as necessary
        layer, layerName, i = getLayer(layer, i)

        # Add the vgg (or modified vgg) layer
        model.add_module(layerName, layer)

        if layerName in contentLayers:
            ll = idx
            addLayer(i, contentImage, model, ContentLoss, 
                     'contentloss', contentLosses)

        elif layerName in styleLayers:
            ll = idx
            addLayer(i, styleImage, model, StyleLoss, 
                     'styleloss', styleLosses)

    model = model[0:idx]

    return model, styleLosses, contentLosses



#inputImage = torch.randn(contentImage.data.size(), device=device)
inputImage = contentImage.clone()

def calculateLoss(epoch, optimizer, inputImage, 
                  model, styleLosses, contentLosses,
                  styleWeight, contentWeight):
    inputImage.data.clamp_(0, 1)

    optimizer.zero_grad()

    model(inputImage)

    styleLoss   = sum([s.loss for s in styleLosses])
    contentLoss = sum([c.loss for c in contentLosses])

    styleLoss *= styleWeight
    contentLoss *= contentWeight

    loss = styleLoss + contentLoss
    loss.backward()

    print('Epoch {}'.format(epoch), 'Style Loss: {:10.2f}'.format(styleLoss.item()), 
          'ContentLoss: {:10.2f}'.format(contentLoss.item()))

    return styleLoss + contentLoss


def run(cnn, normMean, normStd, contentImage, styleImage, 
        inputImage, epochs, styleWeight, contentWeight):

    model, styleLosses, contentLosses = createModel(cnn, normMean, normStd, 
                                                    styleImage, contentImage)

    optimizer = optim.LBFGS([inputImage.requires_grad_()])

    for e in range(epochs):
        optimizer.step(lambda: calculateLoss(
            e, optimizer, inputImage, model, styleLosses, 
            contentLosses, styleWeight, contentWeight))

    inputImage.data.clamp_(0, 1)
    plt.figure()
    plotTensor(inputImage)
    plt.show()


run(cnn, normMean, normStd, contentImage, styleImage, inputImage, 55, 1e9, 1)