import torch.nn as nn
import copy
import torch

# from res_model import resnet50


class ResNormLayer(nn.Module):
    def __init__(self, linear_size, ):
        super(ResNormLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.norm_fn1 = nn.LayerNorm(self.l_size)
        self.norm_fn2 = nn.LayerNorm(self.l_size)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.norm_fn1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        y = self.norm_fn2(y)
        out = x + y
        return out


"""
定义一个不变性编码器
1.全连接层
2.sigmoid激活层
3.dropout层
4。全连接层
5.sigmoid层
6.dropout层
7.全连接层
8.sigmoid层
9.dropout层
10.全连接层
"""


class SharedEncoder(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, output_size=2048, drop_rate=0.):
        super(SharedEncoder, self).__init__()
        self.shared_1 = nn.Linear(input_size, hidden_size)
        self.shared_1_activation = nn.ReLU()
        self.shared_1_dropout = nn.Dropout(drop_rate)

        self.shared_2 = nn.Linear(hidden_size, hidden_size)
        self.shared_2_activation = nn.ReLU()
        self.shared_2_dropout = nn.Dropout(drop_rate)

        self.shared_3 = nn.Linear(hidden_size, hidden_size)
        self.shared_3_activation = nn.ReLU()
        self.shared_3_dropout = nn.Dropout(drop_rate)

        self.shared_4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.shared_1(x)
        x = self.shared_1_activation(x)
        x = self.shared_1_dropout(x)

        x = self.shared_2(x)
        x = self.shared_2_activation(x)
        x = self.shared_2_dropout(x)

        x = self.shared_3(x)
        x = self.shared_3_activation(x)
        x = self.shared_3_dropout(x)

        x = self.shared_4(x)

        return x


"""
基础的自动编码器
这里的自动编码器
自动编码器输入为原始数据
输出为两个：
第一个输出：重建特征
第二个输出：中间层特征
"""


class BaseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed = self.decoder(latent_vector)
        return reconstructed, latent_vector


class ResidualAE(nn.Module):
    ''' Residual autoencoder using fc layers
        layers should be something like [128, 64, 32]
        eg:[128,64,32]-> add: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
                          concat: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
    '''

    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False):
        super(ResidualAE, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.transition = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        for i in range(n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))

    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)

    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer) - 2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i + 1]))
            all_layers.append(nn.ReLU())  # LeakyReLU
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))

        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)

    def forward(self, x, shared):
        x_in = x
        x_out = shared
        latents = []
        for i in range(self.n_blocks):
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            x_in = x_in + shared  # x_out
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return self.transition(x_in + x_out), latents


# class MMIN_model_full(nn.Module):
#     def __init__(self, num_classes=36, meta_dim=1, include_top=True):
#         super(MMIN_model_full, self).__init__()
#
#         self.num_classes = num_classes
#
#         # 构建图像编码器
#         self.image_encoder = resnet50()
#         inchannel = self.image_encoder.fc.in_features
#         self.image_encoder.fc = nn.Linear(inchannel, 36)
#         model_weight_path = "../new_resnet_epoch_99_ckt.pth"
#         self.image_encoder.load_state_dict(torch.load(model_weight_path)['network'])
#
#         # 构建硬度编码器
#         self.meta_head_1 = nn.Sequential(
#             nn.Linear(meta_dim, 2048),
#             nn.ReLU(inplace=True),
#             nn.LayerNorm(2048),
#             ResNormLayer(2048),
#         )
#
#         # 构建一个不变性编码器
#         self.shared_net = SharedEncoder()
#
#         # 定义一个分类器结构，尽量简单
#         self.classifier = nn.Linear(2048 * 4, self.num_classes)
#
#     def forward(self, image, hard):
#         # 计算图像编码
#         images_features = self.image_encoder.conv1(image)
#         images_features = self.image_encoder.bn1(images_features)
#         images_features = self.image_encoder.relu(images_features)
#         images_features = self.image_encoder.maxpool(images_features)
#
#         images_features = self.image_encoder.layer1(images_features)
#         images_features = self.image_encoder.layer2(images_features)
#         images_features = self.image_encoder.layer3(images_features)
#         images_features = self.image_encoder.layer4(images_features)
#
#         images_features = self.image_encoder.avgpool(images_features)
#         images_features = torch.flatten(images_features, 1)
#
#         # 计算硬度编码
#         hardness_features = self.meta_head_1(hard)
#
#         # 通过不变性编码器，这里的不变性编码需要返回并且求解CMD损失
#         Hv = self.shared_net(images_features)
#         Hh = self.shared_net(hardness_features)
#
#         # 计算融合特征，这里的融合方式采用concat
#         h_fused = torch.cat((images_features, hardness_features), dim=1)
#         H_fused = torch.cat((Hv, Hh), dim=1)
#
#         fused = torch.cat((h_fused, H_fused), dim=1)
#
#         # 计算分类结果
#         output = self.classifier(fused)
#
#         # 返回值包括三个：1.分类结果;2.视觉的不变性特征;3.硬度的不变性特征
#         return output, Hv, Hh


class ResidualAE_base(nn.Module):
    ''' Residual autoencoder using fc layers
        layers should be something like [128, 64, 32]
        eg:[128,64,32]-> add: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
                          concat: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
    '''

    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False):
        super(ResidualAE_base, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.transition = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        for i in range(n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))

    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)

    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer) - 2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i + 1]))
            all_layers.append(nn.ReLU())  # LeakyReLU
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))

        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)

    def forward(self, x):
        x_in = x
        x_out = x.clone().fill_(0)
        latents = []
        for i in range(self.n_blocks):
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            x_in = x_in + x_out
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return self.transition(x_in + x_out), latents
