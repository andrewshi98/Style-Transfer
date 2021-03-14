import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Responsible for displaying the end result
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImgUtil():

    @classmethod
    def load_image(cls, file_name, IMG_SIZE):
        # Image size 128.
        trans = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])

        image = Image.open(file_name)
        image = trans(image).unsqueeze(0)
        return image.to(device, torch.float)

# Class for calculating content loss
class ContentLoss(nn.Module):

    def __init__(self, content):
        super(ContentLoss, self).__init__()
        self.content = content.detach()

    def forward(self, img):
        self.loss = F.mse_loss(img, self.content)
        return img

# Class for calculating style loss
class StyleLoss(nn.Module):

    # Remove locational relationship in the style
    @classmethod
    def gram(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d) 
        G = torch.mm(features, features.t()) 
        return G.div(a * b * c * d)

    def __init__(self, style):
        super(StyleLoss, self).__init__()
        self.style = StyleLoss.gram(style).detach()
    
    def forward(self, img):
        self.loss = F.mse_loss(StyleLoss.gram(img), self.style)
        return img

# Class for computing normalized image.
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
    
    def forward(self, img):
        return (img - self.mean) / self.std

class StyleTransfer():
    
    def __init__(self, style_name, content_name, IMG_SIZE=256):
        style_img = ImgUtil.load_image(style_name, IMG_SIZE)
        content_img = ImgUtil.load_image(content_name, IMG_SIZE)

        assert(style_img.shape == content_img.shape)
        self.vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        # Loss calculating layers, defined in paper
        self.content_layers=[4]
        self.style_layers=[1, 2, 3, 4, 5]

        self.__precompute(style_img, content_img)

    # Get image loss model, and loss fetching references
    # Initiating self.model, self.content_losses, self.style_losses
    def __precompute(self, style, content):
        
        self.content_losses = []
        self.style_losses = []

        normalized = Normalization(self.img_mean, self.img_std).to(device)
        self.model = nn.Sequential(normalized)

        # Know where to stop
        end_at = max(max(self.content_layers), max(self.style_layers))

        conv_count = 0
        i = 0
        for layer in self.vgg.children():
            process = False

            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.Conv2d):
                conv_count += 1
                process = True

            self.model.add_module(f"layer_{i}", layer)

            if process:
                # Compute target value for each content, used for calculating losses.
                if conv_count in self.content_layers:
                    content_loss = ContentLoss(self.model(content).detach())
                    self.model.add_module("content_loss_{}".format(conv_count), content_loss)
                    self.content_losses.append(content_loss)

                if conv_count in self.style_layers:
                    style_loss = StyleLoss(self.model(style).detach())
                    self.model.add_module("style_loss_{}".format(conv_count), style_loss)
                    self.style_losses.append(style_loss)

            i += 1
            if conv_count == end_at:
                break

    def style_transfer(self, init_img, num_steps=300,
                        style_weight=800000, content_weight=1, progress = {}):
        optimizer = optim.LBFGS([init_img.requires_grad_()])
        # For closure and multithread purpose.
        refCounter = [0]
        while refCounter[0] <= num_steps:

            def closure():
                init_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                self.model(init_img)
                style_contrib = 0
                content_contrib = 0

                for sl in self.style_losses:
                    style_contrib += sl.loss
                for cl in self.content_losses:
                    content_contrib += cl.loss

                style_contrib *= style_weight
                content_contrib *= content_weight

                loss = style_contrib + content_contrib
                loss.backward()

                refCounter[0] += 1
                progress["value"] = refCounter[0] * 100 / num_steps

                return style_contrib + content_contrib

            optimizer.step(closure)

        init_img.data.clamp_(0, 1)

        return init_img