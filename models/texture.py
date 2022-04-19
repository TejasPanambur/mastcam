import torchvision.models as models
import torch
import torch.nn as nn 
from models.encodinglayer import Encoding, Normalize, View

__all__ = ['TextureArch','texres18']
class TextureArch(nn.Module):
    def __init__(self, nclass, backbone,pretrained=True):
        super(TextureArch, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if self.backbone == 'resnet18':
            self.pretrained = models.resnet18(pretrained=pretrained)
            self.pretrained.fc = None
            enc_in = 512
        elif self.backbone == 'resnet50':
            self.pretrained = models.resnet50(pretrained=pretrained)
            self.pretrained.fc = None
            enc_in = 2048
        elif self.backbone == 'resnet101':
            self.pretrained = models.resnet101(pretrained=pretrained)
            self.pretrained.fc = None
            enc_in = 2048
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))
        n_codes = 8
        self.encoding = nn.Sequential(
            Encoding(D=512,K=n_codes),
            View(-1, 512*n_codes),
            Normalize(),
            nn.Linear(512*n_codes, 64)
        )
        self.fc1_2 = nn.Linear(512, 64)
        self.bilinear = torch.nn.Bilinear(64,64,4096)
        self.embed_layer = nn.Sequential(Normalize(),
                                         nn.Linear(4096, 512))
        

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        fc1_1 = self.encoding(x)
        avg = self.pretrained.avgpool(x).squeeze()
        
        fc1_2 = self.fc1_2(avg)
        x = self.bilinear(fc1_1,fc1_2)
        x = self.embed_layer(x)
        
        return x
    
def texres18(pretrained=True, out=512):
    model = TextureArch(out, 'resnet18',pretrained)
    return model