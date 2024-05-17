import torch
import torchvision
from torch import nn

from s2igan.utils import set_non_grad


class ImageEncoder(nn.Module):
    """
    image encoder pretrained on imagenet
    """

    # def __init__(self, output_dim: int = 1024):
    #     super().__init__()
    #     try:
    #         weights = torchvision.models.get_weight("Inception_V3_Weights.DEFAULT")
    #         model = torchvision.models.inception_v3(weights=weights)
    #     except:
    #         model = torchvision.models.inception_v3(pretrained=True)
    #     model.AuxLogits = None
    #     model.aux_logits = False
    #     set_non_grad(model)
    #     self.model = model
    #     self.model.fc = nn.Linear(2048, output_dim)
    def __init__(self, output_dim: int = 1024):
        super().__init__()
        try:
            weights = torchvision.models.get_weight("VGG16_Weights.DEFAULT")
            model = torchvision.models.vgg16(weights=weights)
        except:
            model = torchvision.models.vgg16(pretrained=True)
        # Remove the classifier layer
        model.classifier = nn.Sequential()
        set_non_grad(model.features)
        self.model = model
        self.model.classifier = nn.Linear(25088, output_dim)  # Adjust the input size according to VGG16

    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def freeze_params(self):
        for p in self.parameters():
            p.requires_grads = False

    def forward(self, img):
        """
        img: (-1, 3, 299, 299)
        out: (-1, output_dim)
        """
        img = nn.functional.interpolate(
            img, size=(299, 299), mode="bilinear", align_corners=False
        )
        out = self.model(img)
        return nn.functional.normalize(out, p=2, dim=1)