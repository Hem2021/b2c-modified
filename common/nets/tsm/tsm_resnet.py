import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights

# Print all available weight versions
# print(ResNet18_Weights.__members__)
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
# from torchvision.models.resnet import model_urls
from nets.tsm.tsm_util import BasicBlock, Bottleneck
from config import cfg
from temp_utilis import logger

class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type, frame_num):
	
        resnet_spec = {
            18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18', resnet18),
            34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34', resnet34),
            50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50', resnet50),
            101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101', resnet101),
            152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152', resnet152),
        }
        block, layers, channels, name, model_constructor = resnet_spec[resnet_type]
        self.frame_num = frame_num
        self.name = name
        self.model_constructor = model_constructor  # Store the ResNet model constructor
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()

        # define the initial layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], frame_num=self.frame_num)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, frame_num=self.frame_num)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, frame_num=self.frame_num)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, frame_num=self.frame_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, frame_num=cfg.frame_per_seg):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.frame_num))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, frame_num=self.frame_num))

        return nn.Sequential(*layers)
    
    def forward(self, x, skip_early):
        if not skip_early:
            # print("inside tsm_resnet forward : ", x.shape) 
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    

    def init_weights(self):
        # Use the constructor to get the pre-trained model weights
        pretrained_resnet = self.model_constructor(weights='DEFAULT').state_dict()

        # Remove final fully connected layer weights, as it's not used
        pretrained_resnet.pop('fc.weight', None)
        pretrained_resnet.pop('fc.bias', None)

        # Load the state dictionary into the backbone
        self.load_state_dict(pretrained_resnet, strict=False)
        print(f"Initialized {self.name} with pre-trained weights")


    # def init_weights(self):
    #     org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
    #     # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
    #     org_resnet.pop('fc.weight', None)
    #     org_resnet.pop('fc.bias', None)
        
    #     self.load_state_dict(org_resnet, strict=False)
    #     print("Initialize resnet from model zoo")


