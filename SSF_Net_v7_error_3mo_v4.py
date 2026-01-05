"""
此时我已经有训练好的resnet单模态模型的权重
我需要结合调制层来训练一个基于硬度的多模态模型
此时可以基于基本上同一个权重来同时进行多模态和单模态的推理

仍然将模态视为统一整体
当模态出现组合缺失时，缺失部分直接使用-1进行填补

444.引入DMIB框架：融合后加入一个瓶颈模块（通过信息瓶颈机制压缩融合后的特征），
最后引入一个充足性损失（确保瓶颈特征中保留尽可能多的task-relevant信息）
该版本中，将直接采用原本的分类头
参数更新的部分：只有瓶颈模块×
更正：参数更新的部分：瓶颈模块+一个错误分类头


注意：v4版本为论文的主要版本，普通v4版本的训练错误率为30%

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets
import os
from torch.optim import *
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchtoolbox.transform import Cutout
from timm.data.mixup import Mixup

from swin_transformer_model import swin_base_patch4_window12_384
from loss_modules import Focal_loss, logit_DKD
from meta_encoder import ResNormLayer
from res_model import resnet50


def setup_seed(seed):
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch的GPU随机种子
    np.random.seed(seed)  # 设置Numpy的随机种子
    random.seed(seed)  # 设置Python的随机种子
    torch.backends.cudnn.deterministic = True  # 设置cudnn的确定性
    torch.backends.cudnn.benchmark = False  # 关闭cudnn的benchmark，以确保每次卷积操作使用相同的算法

candidate_pool = [[2, 1, 1], [1, 0, 1], [1, 1, 0], [2, 1, 0], [3, 1, 0], [2, 0, 1], [3, 0, 1], [1, 1, 1], [3, 1, 1]]

# 36种矿物
classes = ['Agate', 'Albite', 'Almandine', 'Anglesite', 'Azurite', 'Beryl',
           'Cassiterite', 'Chalcopyrite', 'Cinnabar', 'Copper', 'Demantoid', 'Diopside',
           'Elbaite', 'Epidote', 'Fluorite', 'Galena', 'Gold', 'Halite', 'Hematite', 'Magnetite',
           'Malachite', 'Marcasite', 'Opal', 'Orpiment', 'Pyrite', 'Quartz', 'Rhodochrosite', 'Ruby',
           'Sapphire', 'Schorl', 'Sphalerite', 'Stibnite', 'Sulphur', 'Topaz', 'Torbernite', 'Wulfenite']

# 构建一个硬度编号的字典
hardness_dict = {
    'Agate': [3, 1, 0], 'Albite': [3, 1, 0], 'Almandine': [3, 1, 0],
    'Anglesite': [2, 1, 1], 'Azurite': [2, 1, 1], 'Beryl': [3, 1, 0],
    'Cassiterite': [3, 1, 0], 'Chalcopyrite': [2, 0, 1],
    'Cinnabar': [1, 0, 1], 'Copper': [2, 0, 1], 'Demantoid': [3, 1, 0],
    'Diopside': [3, 1, 0], 'Elbaite': [3, 1, 0],
    'Epidote': [3, 1, 1], 'Fluorite': [2, 1, 0], 'Galena': [1, 0, 1],
    'Gold': [2, 0, 1], 'Halite': [1, 1, 0], 'Hematite': [3, 0, 1],
    'Magnetite': [3, 0, 1], 'Malachite': [2, 1, 1],
    'Marcasite': [3, 0, 1], 'Opal': [3, 1, 0],
    'Orpiment': [1, 1, 1], 'Pyrite': [3, 0, 1], 'Quartz': [3, 1, 0],
    'Rhodochrosite': [2, 1, 0], 'Ruby': [3, 1, 0], 'Sapphire': [3, 1, 0],
    'Schorl': [3, 1, 1], 'Sphalerite': [2, 1, 1], 'Stibnite': [1, 0, 1],
    'Sulphur': [1, 1, 1], 'Topaz': [3, 1, 0], 'Torbernite': [1, 1, 1],
    'Wulfenite': [2, 1, 0]}


# 定义一个信息瓶颈层
class InfoBottleneck(nn.Module):
    def __init__(self, input_dim=2084, bottleneck_dim=256):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.down(x)
        f_star = self.up(z)
        return f_star


class SSFLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))  # γ
        self.shift = nn.Parameter(torch.zeros(feature_dim))  # β

    def forward(self, x):
        # x: [B, N, D] 或 [B, D, H, W]
        if x.dim() == 4:
            B, D, H, W = x.shape
            x = x.view(B, D, H * W).permute(0, 2, 1)  # 转换为 [B, N, D]
        x = x * self.scale + self.shift  # 应用缩放和位移
        if x.dim() == 3:
            x = x.permute(0, 2, 1).view(B, D, H, W)  # 恢复卷积层形状
        return x


# 构建一个完整模态的多模态模型 -- 教师模型
class multi_swin_v1_full(nn.Module):
    def __init__(self, num_classes=36, meta_dim=3):
        super(multi_swin_v1_full, self).__init__()
        self.num_classes = num_classes
        self.meta_dim = meta_dim

        # 构建图像编码器
        self.image_encoder = swin_base_patch4_window12_384()
        in_channels = self.image_encoder.head.in_features
        self.image_encoder.head = nn.Linear(in_channels, self.num_classes)
        model_weight = "swin_base_patch4_window12_384_epoch_29_ckt.pth"
        self.image_encoder.load_state_dict(torch.load(model_weight)['network'])

        # 构建多模态编码器
        self.meta_head_1 = nn.Sequential(
            nn.Linear(self.meta_dim, self.num_classes),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.num_classes),
            ResNormLayer(self.num_classes),
        )
        # 构建分类头
        self.classifier = nn.Linear(1024 + self.num_classes, num_classes)

    def get_img_features(self, x):
        # x: [B, L, C]
        x, H, W = self.image_encoder.patch_embed(x)
        x = self.image_encoder.pos_drop(x)

        for layer in self.image_encoder.layers:
            x, H, W = layer(x, H, W)

        x = self.image_encoder.norm(x)  # [B, L, C]
        x = self.image_encoder.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)

        return x

    def get_hard_features(self, hardness):
        hardness_features = self.meta_head_1(hardness)
        return hardness_features

    def get_cls(self, fused):
        output = self.classifier(fused)
        return output

    def forward(self, x, hardness):
        # 图像编码最后输出维度为：[B, 2048]
        # 输出图像编码
        images_features = self.get_img_features(x)

        # 硬度编码最后输出维度为：[B, 2048]
        hardness_features = self.get_hard_features(hardness)

        # 特征编码最后输出维度为：[B, 36]
        fused = torch.cat((images_features, hardness_features), dim=1)
        # 进行分类
        output = self.get_cls(fused)
        return output, images_features, hardness_features


# 构建一个多模态学生模型，其中将引入调制层
# 对于resnet50来说，我暂时只需要引入八个调制层
# 默认情况是缺失 miss参数为1
# 特殊情况为不缺失 miss参数为0
class multi_student_res(nn.Module):
    def __init__(self, num_classes=36, meta_dim=3):
        super().__init__()
        self.num_classes = num_classes
        self.meta_dim = meta_dim

        # 构建图像编码器，图像编码器全权负责单模态的情况
        self.image_encoder = resnet50()
        in_channels = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Linear(in_channels, self.num_classes)
        model_weight = "student_swinbase_epoch_8_ckt.pth"
        self.image_encoder.load_state_dict(torch.load(model_weight)['network'])

        self._freeze_image_encoder()

        # 构建模态编码器，模态编码器
        self.meta_head_1 = nn.Sequential(
            nn.Linear(self.meta_dim, self.num_classes),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.num_classes),
            ResNormLayer(self.num_classes),
        )

        # 构建八个调制层
        self.ssf_layer_1 = SSFLayer(feature_dim=64)
        self.ssf_layer_2 = SSFLayer(feature_dim=64)
        self.ssf_layer_3 = SSFLayer(feature_dim=64)
        self.ssf_layer_4 = SSFLayer(feature_dim=64)
        self.ssf_layer_5 = SSFLayer(feature_dim=256)
        self.ssf_layer_6 = SSFLayer(feature_dim=512)
        self.ssf_layer_7 = SSFLayer(feature_dim=1024)
        self.ssf_layer_8 = SSFLayer(feature_dim=2048)

        # 构建分类头
        self.classifier_3mo = nn.Linear(2048 + self.num_classes, num_classes)
        self.classifier_2mo_12 = nn.Linear(2048 + self.num_classes, num_classes)
        self.classifier_2mo_13 = nn.Linear(2048 + self.num_classes, num_classes)
        self.classifier_2mo_23 = nn.Linear(2048 + self.num_classes, num_classes)
        self.classifier_1mo_1 = nn.Linear(2048 + self.num_classes, num_classes)
        self.classifier_1mo_2 = nn.Linear(2048 + self.num_classes, num_classes)
        self.classifier_1mo_3 = nn.Linear(2048 + self.num_classes, num_classes)

        # 定义一个错误条件下的分类头
        self.classifier_3mo_error = nn.Linear(2048 + self.num_classes, num_classes)

        # 定义一个信息瓶颈层
        self.info_layer = InfoBottleneck(2048 + self.num_classes, 256)

    def get_ssf_img_features(self, x):
        with torch.no_grad():
            x = self.image_encoder.conv1(x)
            x = self.ssf_layer_1(x)

            x = self.image_encoder.bn1(x)
            x = self.ssf_layer_2(x)

            x = self.image_encoder.relu(x)
            x = self.ssf_layer_3(x)

            x = self.image_encoder.maxpool(x)
            x = self.ssf_layer_4(x)

            x = self.image_encoder.layer1(x)
            x = self.ssf_layer_5(x)

            x = self.image_encoder.layer2(x)
            x = self.ssf_layer_6(x)

            x = self.image_encoder.layer3(x)
            x = self.ssf_layer_7(x)

            x = self.image_encoder.layer4(x)
            x = self.ssf_layer_8(x)

            x = self.image_encoder.avgpool(x)
            x = torch.flatten(x, 1)
        return x

    def get_img_features(self, x):
        x = self.image_encoder.conv1(x)

        x = self.image_encoder.bn1(x)
        x = self.image_encoder.relu(x)
        x = self.image_encoder.maxpool(x)

        x = self.image_encoder.layer1(x)
        x = self.image_encoder.layer2(x)
        x = self.image_encoder.layer3(x)
        x = self.image_encoder.layer4(x)

        x = self.image_encoder.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_hard_featrues(self, x):
        hardness_features = self.meta_head_1(x)
        return hardness_features

    def get_single_fc(self, x):
        x = self.image_encoder.conv1(x)
        x = self.image_encoder.bn1(x)
        x = self.image_encoder.relu(x)
        x = self.image_encoder.maxpool(x)

        x = self.image_encoder.layer1(x)
        x = self.image_encoder.layer2(x)
        x = self.image_encoder.layer3(x)
        x = self.image_encoder.layer4(x)

        x = self.image_encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.image_encoder.fc(x)
        return x

    def get_multi_fc(self, x, hard, miss):
        image_features = self.get_ssf_img_features(x)
        hardness_features = self.get_hard_featrues(hard)

        fused = torch.cat((image_features, hardness_features), dim=1)

        # 经过信息瓶颈层
        fused_info = self.info_layer(fused)


        if miss == 0:
            cls = self.classifier_3mo(fused)
        elif miss == 2:
            cls = self.classifier_2mo_12(fused)
        elif miss == 3:
            cls = self.classifier_2mo_13(fused)
        elif miss == 4:
            cls = self.classifier_2mo_23(fused)
        elif miss == 5:
            cls = self.classifier_1mo_1(fused)
        elif miss == 6:
            cls = self.classifier_1mo_2(fused)
        elif miss == 7:
            cls = self.classifier_1mo_3(fused)
        elif miss == 10:
            cls_info = self.classifier_3mo_error(fused_info)
            cls = self.classifier_3mo(fused)

        else:
            cls = self.classifier_3mo(fused)
        # cls = self.classifier(fused)
        return cls, cls_info

    def _freeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False  # 确保冻结

        self.image_encoder.eval()

    def forward(self, x, hard, miss):
        if miss == 1:  # 模态完全缺失的情况
            output = self.get_single_fc(x)
        elif miss == 0:  # 三模态 模态完整的情况
            output, gate, hard_unc = self.get_multi_fc(x, hard, miss)
        elif miss == 2:  # 双模态 硬度光泽的情况
            output = self.get_multi_fc(x, hard, miss)
        elif miss == 3:  # 双模态 硬度条痕的情况
            output = self.get_multi_fc(x, hard, miss)
        elif miss == 4:  # 双模态 光泽条痕的情况
            output = self.get_multi_fc(x, hard, miss)
        elif miss == 5:  # 单模态 硬度的情况
            output = self.get_multi_fc(x, hard, miss)
        elif miss == 6:  # 单模态 光泽的情况
            output = self.get_multi_fc(x, hard, miss)
        elif miss == 7:  # 单模态 条痕的情况
            output = self.get_multi_fc(x, hard, miss)
        elif miss == 10:  # 训练含错误模态的情况
            output = self.get_multi_fc(x, hard, miss)
        else:
            output = self.get_single_fc(x)
        return output


# 训练设备定义
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logger.info(f"device : {device}")

# 图像预处理
train_preprocess = transforms.Compose([
    transforms.RandomResizedCrop(384),
    transforms.RandomHorizontalFlip(),
    # Cutout(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_preprocess = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class MyDataset(Dataset):
    def __init__(self, img_root, meta_root, is_train):
        # 1.图片根目录
        self.img_root = img_root
        self.meta_root = meta_root
        # 2.训练图片和验证图片地址(根据自己的情况更改)
        self.train_set_file = os.path.join(meta_root, 'train.txt')
        self.val_set_file = os.path.join(meta_root, 'val.txt')
        # 3.模型用于训练或是验证
        self.is_train = is_train
        # 5.获得数据
        # 图片实例
        self.samples = []
        # 标签实例
        self.sam_labels = []
        # 标签对应的硬度实例
        self.sam_hardness = []
        # 5.1 训练还是验证数据集
        self.read_file = ""
        self.is_train = is_train
        if self.is_train:
            self.read_file = self.train_set_file
        else:
            self.read_file = self.val_set_file
        # 5.2 获得所有的样本(根据自己的情况更改)
        with open(self.read_file, 'r') as f:
            for line in f:
                img_path = os.path.join(self.img_root, line.strip().split('/')[0] + '.jpg')
                label = line.strip().split('/')[1]
                # label = label.replace("_", " ")
                label_text = label
                # 将图片放入sample
                self.samples.append(img_path)
                # 将label放入sample
                self.sam_labels.append(label_text)
                # 根据标签名称，随机取一个莫氏硬度值
                # 取得一个数组
                hardness = hardness_dict[label]
                # test_hardness = random.choice(hardness)
                self.sam_hardness.append(hardness)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.sam_labels[idx]
        hardness_value = self.sam_hardness[idx]
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        if self.is_train:
            image_tensor = train_preprocess(image)
        else:
            image_tensor = val_preprocess(image)
        target_label_index = classes.index(label)
        label_list = np.zeros(36)
        label_list[target_label_index] = 1

        hardness_value_list = [hardness_value]
        hardness_value_list = np.array(hardness_value_list)
        hardness_value_list = torch.from_numpy(hardness_value_list).float()
        return image_tensor, label_list, hardness_value_list


root = "/root/autodl-tmp"

# 加载数据集 - 训练集
my_dataset = MyDataset(img_root='/root/autodl-tmp/images',
                       meta_root='/root/autodl-tmp/meta',
                       is_train=True)

dataset_size_of_mine = len(my_dataset)
# 此处的batch_size设置为1 否则会报错
my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True, pin_memory=False)

# 加载数据集 - 测试集
my_dataset_val = MyDataset(img_root='/root/autodl-tmp/images',
                           meta_root='/root/autodl-tmp/meta',
                           is_train=False)

dataset_size_of_mine_val = len(my_dataset_val)
# 此处的batch_size设置为1 否则会报错
my_dataloader_val = DataLoader(my_dataset_val, batch_size=64, shuffle=True, pin_memory=False)

# 定义模型
# 请注意，当参数miss=0时，模型调用多模态状态
model = multi_student_res()
model_weight = "SSF_Net_v6_3mo_epoch_29_ckt.pth"
model.load_state_dict(torch.load(model_weight, map_location=device)['network'], strict=False)
model.to(device)

# for name, param in model.named_parameters():
#     print(f"{name}: {param.requires_grad}")  # 应全部输出 False

# 定义教师模型
# teacher_model = multi_swin_v1_full()
# teacher_model_weight = "teacher_3mo_epoch_5_ckt.pth"
# teacher_model.load_state_dict(torch.load(teacher_model_weight, map_location=device)['network'])
# teacher_model.to(device)

model_name = "SSF_Net_v7_error_found_3mo_v4_v2"

# 定义损失函数
criterion_cls = nn.CrossEntropyLoss()
# 定义蒸馏损失函数
criterion_kd = logit_DKD()

# 初始学习率 0.01
learning_rate = 0.01
# 最大学习轮次
epoches = 100
# optimizer = SGD([
#     {'params': model.ssf_layer_1.parameters()},
#     {'params': model.ssf_layer_2.parameters()},
#     {'params': model.ssf_layer_3.parameters()},
#     {'params': model.ssf_layer_4.parameters()},
#     {'params': model.ssf_layer_5.parameters()},
#     {'params': model.ssf_layer_6.parameters()},
#     {'params': model.ssf_layer_7.parameters()},
#     {'params': model.ssf_layer_8.parameters()},
#     {'params': model.meta_head_1.parameters()},
#     {'params': model.classifier_3mo.parameters()},
#     {'params': model.umoe.parameters()},
#     {'params': model.hard_unc_net.parameters()},
# ], lr=learning_rate, weight_decay=0.0005)
optimizer = SGD([
    {'params': model.info_layer.parameters()},
    {'params': model.classifier_3mo_error.parameters()},
], lr=learning_rate, weight_decay=0.0005)

# 定义余弦降低学习率
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=2e-8)

save_gap = 5  # 每15轮保存一次权重参数
ckt_gap = 1  # 每3轮验证验证集一次

# 样本错误值生成器
def batch_constrained_corruption(hardness_token, candidate_pool, error_rate=1.0):
    """
    为每个样本生成错误值，确保新值≠原值且来自候选池
    Args:
        hardness_token: 原始张量 (batch_size, 3)
        candidate_pool: 所有候选值的数组 (n, 3)
        error_rate: 错误比例 (0~1)
    Returns:
        corrupted_token: 替换后的张量
    """
    device = hardness_token.device
    batch_size = hardness_token.size(0)
    n_errors = int(batch_size * error_rate)

    if n_errors == 0:
        return hardness_token

    # 1. 将候选池转为张量并移至相同设备
    candidate_tensor = torch.tensor(candidate_pool, dtype=hardness_token.dtype, device=device)

    # 2. 随机选择要替换的样本索引
    error_indices = torch.randperm(batch_size)[:n_errors].to(device)

    # 3. 为每个错误样本生成专属候选池（排除自身值）
    corrupted_token = hardness_token.clone()
    for idx in error_indices:
        original_val = hardness_token[idx]
        # 构建专属候选池：排除自身值
        mask = ~(candidate_tensor == original_val).all(dim=1)
        valid_candidates = candidate_tensor[mask]
        # 随机选择新值
        replace_idx = torch.randint(0, len(valid_candidates), (1,))
        corrupted_token[idx] = valid_candidates[replace_idx]

    return corrupted_token


# 定义验证的函数
def evaluate(model, criterion):
    model.eval()
    corrects = eval_loss = 0
    eval_loss_info = 0
    corrects_info = 0

    top1_correct = 0
    top5_correct = 0

    top5_correct_info = 0
    total = 0

    with torch.no_grad():
        # for image, label, hardness_token in tqdm(my_dataloader_val):
        for image, label, hardness_token in my_dataloader_val:
            image = image.to(device)
            label = label.to(device)
            hardness_token = hardness_token.to(device)
            hardness_token = hardness_token.squeeze(1)

            labels = torch.argmax(label, dim=1)

            pred, pred_info = model(image, hardness_token, 10)

            # 计算相关性分支的验证值
            pred_info = torch.softmax(pred_info, dim=1)
            loss_info = criterion(pred_info, label)
            eval_loss_info += loss_info.item()
            max_value_info, max_index_info = torch.max(pred_info, 1)
            _, max_index_true_label = torch.max(label, 1)
            pred_label_info = max_index_info.cpu().numpy()
            true_label = max_index_true_label.cpu().numpy()
            corrects_info += np.sum(pred_label_info == true_label)

            _, top5_pred_info = pred_info.topk(5, dim=1, largest=True, sorted=True)
            top5_correct_info += (top5_pred_info == labels.unsqueeze(1)).sum().item()


            pred = torch.softmax(pred, dim=1)
            loss = criterion(pred, label)
            eval_loss += loss.item()
            max_value, max_index = torch.max(pred, 1)
            _, max_index_true_label = torch.max(label, 1)
            pred_label = max_index.cpu().numpy()
            true_label = max_index_true_label.cpu().numpy()
            corrects += np.sum(pred_label == true_label)

            _, top5_pred = pred.topk(5, dim=1, largest=True, sorted=True)
            top5_correct += (top5_pred == labels.unsqueeze(1)).sum().item()

    val_sum = dataset_size_of_mine_val
    val_acc = corrects / val_sum
    val_acc_num = corrects
    eval_loss = eval_loss / float(val_sum)

    top5_accuracy = 100 * top5_correct / val_sum

    logger.info(f"模型信息瓶颈层top1精度为：{corrects_info / val_sum}%")
    logger.info(f"模型信息瓶颈层损失值为：{eval_loss_info / float(val_sum)}")
    logger.info(f"模型信息瓶颈层top5精度为：{100 * top5_correct_info / val_sum}%")

    logger.info(f"模型top5精度为：{top5_accuracy}%")
    # 返回验证损失函数 验证预测正确的个数 验证预测正确的比率
    return eval_loss, val_acc_num, val_acc


"""
2025-02-19 15:08:54.899 | INFO     | __main__:<module>:159 - =============================================================
2025-02-19 15:08:54.969 | INFO     | __main__:<module>:163 - 1.维度为：torch.Size([2, 64, 192, 192])
2025-02-19 15:08:54.991 | INFO     | __main__:<module>:166 - 2.维度为：torch.Size([2, 64, 192, 192])
2025-02-19 15:08:54.999 | INFO     | __main__:<module>:169 - 3.维度为：torch.Size([2, 64, 192, 192])
2025-02-19 15:08:55.033 | INFO     | __main__:<module>:172 - 4.维度为：torch.Size([2, 64, 96, 96])
2025-02-19 15:08:55.208 | INFO     | __main__:<module>:176 - 5.维度为：torch.Size([2, 256, 96, 96])
2025-02-19 15:08:55.373 | INFO     | __main__:<module>:179 - 6.维度为：torch.Size([2, 512, 48, 48])
2025-02-19 15:08:55.539 | INFO     | __main__:<module>:182 - 7.维度为：torch.Size([2, 1024, 24, 24])
2025-02-19 15:08:55.619 | INFO     | __main__:<module>:185 - 8.维度为：torch.Size([2, 2048, 12, 12])
2025-02-19 15:08:55.643 | INFO     | __main__:<module>:188 - 9.维度为：torch.Size([2, 2048, 1, 1])
2025-02-19 15:08:55.652 | INFO     | __main__:<module>:191 - 10.维度为：torch.Size([2, 2048])
2025-02-19 15:08:55.682 | INFO     | __main__:<module>:194 - 11.维度为：torch.Size([2, 1000])
Process finished with exit code 0
"""
if __name__ == '__main__':
    # eval_loss, val_acc_num, val_acc = evaluate(model, criterion_cls)
    # logger.info(f"验证损失：{eval_loss} 验证正确数：{val_acc_num} 验证准确率：{val_acc}")
    save_index = 1
    logger.info(f"{model_name}训练开始")
    # 定义一个writer
    # writer = SummaryWriter()

    no_improve_epochs = 0  # 记录未提升的epoch

    for epoch in range(epoches):
        # 定义模型训练
        model.train()
        # 定义教师模型测试
        # teacher_model.eval()
        # 定义总学习率
        total_loss = 0
        # 定义总准确个数
        train_corrects = 0

        for image, label, hardness_token in my_dataloader:
            image = image.to(device)
            label = label.to(device)

            hardness_token = hardness_token.to(device)
            hardness_token = hardness_token.squeeze(1)

            corrupted_hardness_token = batch_constrained_corruption(
                hardness_token,
                candidate_pool=candidate_pool,
                error_rate=0.3
            )

            # logger.info(f"hardness_token:{hardness_token[0]}")
            # logger.info(f"hard_noisy:{hard_noisy[0]}")

            class_indices = torch.argmax(label, dim=1)

            # with torch.no_grad():
            #     multi_output, _, _ = teacher_model(image, hardness_token)

            student_output, student_output_info = model(image, corrupted_hardness_token, 10)
            student_target = torch.softmax(student_output, dim=1)
            # 计算硬标签损失
            hard_loss = criterion_cls(student_output_info, label)
            # 计算相关性损失
            p_origin = F.softmax(student_output)
            p_info = F.log_softmax(student_output_info)
            soft_loss = F.kl_div(p_info, p_origin, reduction='batchmean')

            # # 计算软标签损失
            # soft_loss = criterion_kd(student_output, multi_output, class_indices)

            loss = hard_loss + soft_loss
            # logger.info(f"hard_loss:{hard_loss}")
            # logger.info(f"soft_loss:{soft_loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            max_value, max_index = torch.max(student_output, 1)
            _, max_index_true_label = torch.max(label, 1)
            pred_label = max_index.cpu().numpy()
            true_label = max_index_true_label.cpu().numpy()

            train_corrects += np.sum(pred_label == true_label)

        scheduler.step()

        epoch_loss = total_loss / float(dataset_size_of_mine)
        acc = train_corrects / float(dataset_size_of_mine) * 100.0
        # 显示每一轮的准确率 和 损失函数
        logger.info('epoch: {} Train_Accuracy: {}% Train_Loss: {:.4f}'.format(epoch, acc, epoch_loss))

        if save_index % ckt_gap == 0 or save_index == 1:
            eval_loss, val_acc_num, val_acc = evaluate(model, criterion_cls)
            logger.info(f"验证损失：{eval_loss} 验证正确数：{val_acc_num} 验证准确率：{val_acc}")

        if save_index % save_gap == 0:
            checkpoint_path = f"{model_name}_epoch_{epoch}_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            # 保存模型的部分参数
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"{model_name}_checkpoint_{epoch} saved")
        save_index += 1

    # writer.close()
    logger.info(f"{model_name}训练结束")
