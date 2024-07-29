"""
    implementation of other two-way contrastive losses
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    # 从多个显卡进程中获取tensor并聚合在一起
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        input = input.contiguous()
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CONT_Loss(nn.Module):
    def __init__(
        self,
        world_size=8,
        temperature=0.01,
        learnable_temp=False,
    ):
        super(CONT_Loss, self).__init__()
        self.world_size = world_size
        if learnable_temp:
            self.temperature = nn.Parameter(torch.ones([]) * temperature)
        else:
            self.temperature = torch.ones([]) * temperature

    def forward(self, ts_features, note_features):
        # ts_features  B C D
        # note_features B C D

        # 1. 将多卡上的聚合起来
        # print("con loss 1...")
        if self.world_size > 1:
            ts_features = torch.cat(GatherLayer.apply(ts_features), dim=0)
            note_features = torch.cat(GatherLayer.apply(note_features), dim=0)

        B, C, D = ts_features.shape

        # 2. 维度合并
        # print("con loss 2...")
        ts_features = ts_features.reshape((B*C, D))
        note_features = note_features.reshape((B * C, D))

        # 3. 设置温度超参数
        # print("con loss 3...")
        self.temperature.data = torch.clamp(self.temperature, min=0.01)

        # 4. 计算2种模态数据之间的相似度
        # print("con loss 4...")
        sim = torch.einsum("i d, j d -> i j", ts_features, note_features)

        if torch.isnan(sim).any().item():
            print("sim has nan")

        # 5. 排除同一个病人的表示; 同一个病人的之间的相似度负无穷大即可，这样计算的loss的过程中，就可以排除这部分的影响，使得经过softmax之后的sim为0
        # print("con loss 5...")
        sim_new = sim.clone()
        now_index = 0
        while now_index < B * C:
            sim_new[now_index:now_index+C, now_index:now_index+C] = torch.tensor(-1e4, dtype=torch.float16)  # float('-inf')
            index_2 = now_index
            while index_2 < now_index + C:
                sim_new[index_2, index_2] = sim[index_2, index_2]
                index_2 += 1
            now_index += C
        del sim

        if torch.isnan(sim_new).any().item():
            print("sim new has nan")

        # 6. 利用cross entropy实现对比学习计算
        # print("con loss 6...")
        labels = torch.arange(ts_features.shape[0], device=ts_features.device)
        total_loss = (
            F.cross_entropy(sim_new / self.temperature, labels)
            + F.cross_entropy(sim_new.t() / self.temperature, labels)
        ) / 2

        print("con loss:", total_loss)
        # 替换NaN或Inf为0
        # print("con loss 7...")
        total_loss[torch.isnan(total_loss)] = 0  # 将NaN替换为0  # TODO:研究为什么有nan
        total_loss[torch.isinf(total_loss)] = 0  # 将Inf替换为0

        del sim_new, ts_features, note_features, labels

        # print("con loss 8...")
        return total_loss


class RESTORE_LOSS(nn.Module):
    def __init__(self):
        super(RESTORE_LOSS, self).__init__()

    def forward(self, ts_rep, ts_label, ts_mask, note_rep, note_label):
        # ts_rep  B l K
        # ts_label  B l K
        # ts_mask  B l K
        # note_rep  B D
        # note_label  B D

        # 计算ts的mse

        ts_mse = torch.sum(((ts_rep - ts_label) ** 2) * ts_mask, dim=(1, 2)) / torch.sum(ts_mask, dim=(1, 2))
        ts_mse = torch.mean(ts_mse)

        print("ts_mse:", ts_mse)
        if torch.isnan(ts_mse):
            print("ts_mse has nan!!!!")
            print("ts_rep:", ts_rep)
            print("ts_label:", ts_label)
            print("ts_mask:", torch.sum(ts_mask, dim=(1, 2)))

        # 将note label清掉梯度并修改note rep 的shape为B D
        note_label.detach_()
        B, D = note_label.shape
        note_rep = note_rep.reshape(B, D)

        # 计算note的cos相似度
        note_cos = self.cosine_similarity(note_rep, note_label)
        note_cos = torch.mean(note_cos)

        print("note_cos:", note_cos)

        return ts_mse, note_cos

    def cosine_similarity(self, note_rep, label):
        # 计算内积
        dot_product = torch.sum(note_rep * label, dim=1)

        # 计算范数
        norm_note_rep = torch.norm(note_rep, dim=1)
        norm_label = torch.norm(label, dim=1)

        # 计算余弦相似度
        similarity = dot_product / (norm_note_rep * norm_label)

        # 替换NaN或Inf为0
        similarity[torch.isnan(similarity)] = 0  # 将NaN替换为0
        similarity[torch.isinf(similarity)] = 0  # 将Inf替换为0

        # 取绝对值
        similarity = torch.abs(similarity)
        return similarity