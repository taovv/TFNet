from mmseg.models.backbones.resnet import ResNet
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
class ResNetMoreFeature(ResNet):

    def __init__(self, out_channels=(3, 64, 256, 512, 1024, 2048), **kwargs):
        super(ResNetMoreFeature, self).__init__(**kwargs)
        self.out_channels = out_channels

    def forward(self, x):
        outs = [x]
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs.append(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class ResNetV1cMoreFeature(ResNetMoreFeature):

    def __init__(self, out_channels=(3, 64, 256, 512, 1024, 2048), **kwargs):
        super(ResNetV1cMoreFeature, self).__init__(out_channels, deep_stem=True, avg_down=False, **kwargs)


@BACKBONES.register_module()
class ResNetV1dMoreFeature(ResNetMoreFeature):

    def __init__(self, out_channels=(3, 64, 256, 512, 1024, 2048), **kwargs):
        super(ResNetV1dMoreFeature, self).__init__(out_channels, deep_stem=True, avg_down=True, **kwargs)
