import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Focal_loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        :param preds: softmax输出结果   [batch_size, 36]
        :param labels:真实值   [batch_size, 36]
        :return:损失值
        """
        eps = 1e-7
        y_pred = preds
        y_true = labels

        ce = -1 * torch.log(y_pred + eps) * y_true
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)


# cmd中心距损失函数
class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def forward(self, source, target):

        batch_size = int(source.size()[0])

        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        kernels = sum(kernel_val)  # /len(kernel_val)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class JDA(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mmd_loss = MMD_loss()

    def forward(self, feature_student, feature_teacher, logit_student, logit_teacher):
        # calculate D1(X_u,X_v)
        # 这里将教师模型和学生模型的特征map拉伸后进行mmd损失计算（最大均值差异）
        feature_student = feature_student.view(feature_student.shape[0], -1)
        feature_teacher = feature_teacher.view(feature_teacher.shape[0], -1)
        D1 = self.mmd_loss(feature_student, feature_teacher)

        # calculate D2(Y|X_u,Y|X_v)
        T = 2
        D2 = F.kl_div(F.log_softmax(logit_student / T, dim=1),
                      F.softmax(logit_teacher / T, dim=1),
                      reduction='batchmean') * T * T

        jda_loss = D1 * self.alpha + D2 * self.beta

        return jda_loss


"""
alpha:给定每个种类的重要性
alpha = [0.01712451 0.04637763 0.02375785 0.03179529 0.01055296 0.00624109
 0.02544132 0.03627042 0.1141209  0.00988076 0.0568579  0.02858164
 0.01156091 0.01386306 0.00304502 0.015173   0.02144585 0.05950489
 0.01699699 0.04664845 0.01420193 0.0488596  0.01620403 0.06086499
 0.00794885 0.00139545 0.01459103 0.05088243 0.03999277 0.02847975
 0.00985942 0.01863226 0.02229889 0.01369081 0.05112667 0.0057307 ]
"""


# # 训练设备定义
# # device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None,
#                  gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             alpha = [0.01712451, 0.04637763, 0.02375785, 0.03179529, 0.01055296, 0.00624109,
#                      0.02544132, 0.03627042, 0.1141209, 0.00988076, 0.0568579, 0.02858164,
#                      0.01156091, 0.01386306, 0.00304502, 0.015173, 0.02144585, 0.05950489,
#                      0.01699699, 0.04664845, 0.01420193, 0.0488596, 0.01620403, 0.06086499,
#                      0.00794885, 0.00139545, 0.01459103, 0.05088243, 0.03999277, 0.02847975,
#                      0.00985942, 0.01863226, 0.02229889, 0.01369081, 0.05112667, 0.0057307]
#         self.alpha = torch.tensor(alpha).to(device)
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         # 计算 softmax
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#
#         if self.reduction == 'mean':
#             return torch.mean(F_loss)
#         elif self.reduction == 'sum':
#             return torch.sum(F_loss)
#         else:
#             return F_loss


# 构建RKD蒸馏损失
class RKDLoss(nn.Module):
    """关系知识蒸馏，CVPR2019"""

    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)  # 将学生特征展平
        teacher = f_t.view(f_t.shape[0], -1)  # 将教师特征展平
        # RKD距离损失
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)  # 计算教师特征之间的距离
            mean_td = t_d[t_d > 0].mean()  # 计算非零距离的平均值
            t_d = t_d / mean_td  # 标准化距离
        d = self.pdist(student, squared=False)  # 计算学生特征之间的距离
        mean_d = d[d > 0].mean()  # 计算非零距离的平均值
        d = d / mean_d  # 标准化距离
        loss_d = F.smooth_l1_loss(d, t_d)  # 计算距离损失
        # RKD角度损失
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))  # 计算教师特征之间的差值
            norm_td = F.normalize(td, p=2, dim=2)  # 归一化教师特征之间的差值
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)  # 计算教师特征之间的角度
        sd = (student.unsqueeze(0) - student.unsqueeze(1))  # 计算学生特征之间的差值
        norm_sd = F.normalize(sd, p=2, dim=2)  # 归一化学生特征之间的差值
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)  # 计算学生特征之间的角度
        loss_a = F.smooth_l1_loss(s_angle, t_angle)  # 计算角度损失
        loss = self.w_d * loss_d + self.w_a * loss_a  # 综合距禋损失和角度损失
        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)  # 计算特征的平方和
        prod = e @ e.t()  # 计算特征之间的乘积
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)  # 计算特征之间的欧氏距离
        if not squared:
            res = res.sqrt()  # 如果不是平方距离，则开方得到欧氏距离
        res = res.clone()  # 克隆结果
        res[range(len(e)), range(len(e))] = 0  # 对角线元素置零
        return res


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKDloss(nn.Module):
    def __init__(self):
        super(DKDloss, self).__init__()

    def forward(self, logits_student, logits_teacher, target, alpha=1.0, beta=8.0, temperature=3.0):
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, reduction='sum')
                * (temperature ** 2) / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
                * (temperature ** 2) / target.shape[0]
        )

        loss = alpha * tckd_loss + beta * nckd_loss

        return loss

class logit_DKD(nn.Module):
    def __init__(self):
        super(logit_DKD, self).__init__()

    def forward(self, logits_student, logits_teacher, target, alpha=1.0, beta=8.0, temperature=3.0, logit_stand=True):
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        logits_student = normalize(logits_student) if logit_stand else logits_student
        logits_teacher = normalize(logits_teacher) if logit_stand else logits_teacher
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, reduction='sum')
                * (temperature ** 2) / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
                * (temperature ** 2) / target.shape[0]
        )

        loss = alpha * tckd_loss + beta * nckd_loss

        return loss


# logits标准化
def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


# kd蒸馏损失--无标准化（类形式）
class DistillKL(nn.Module):
    def __init__(self, T=3.0):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


# kd蒸馏损失--带标准化(函数形式)
def kd_loss(logits_student_in, logits_teacher_in, temperature=3.0, logit_stand=True):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd

class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5):
        '''
        bins: split to n bins
        alpha: hyper-parameter
        '''
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    '''
        GHM_Loss for classification
    '''

    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMR_Loss(GHM_Loss):
    '''
        GHM_Loss for regression
    '''

    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)


class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 reduction: str = "none",
                 weight: Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        在这里 理论上epsilon参数需要经过实验进行验证，根据论文里的研究3-5已经到了最好的情况，而默认参数为1，在我的实验中，epsilon取5
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        # labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
        #                                                                    dtype=logits.dtype)


        pt = torch.sum(labels * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1
