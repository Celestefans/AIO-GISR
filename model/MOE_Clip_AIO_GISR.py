import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange 
from torch.distributions.normal import Normal
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_feat, h_feat=None, out_feat=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_feat = out_feat or in_feat
        h_feat = h_feat or in_feat
        self.fc1 = nn.Linear(in_feat, h_feat)
        self.act = act_layer()
        self.fc2 = nn.Linear(h_feat, out_feat)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, mlp_ratio, num_experts, noisy_gating=True, use_experts=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = use_experts
        # instantiate experts
        self.experts = nn.ModuleList([Mlp(input_size, h_feat=int(input_size*mlp_ratio), out_feat=output_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.randn(2*input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(2*input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.conv1 = nn.Conv2d(self.input_size * 2, self.input_size * 2, 1, 1)
        self.conv2 = nn.Conv2d(self.input_size * 3, self.input_size * 2, 1, 1)
        self.conv3 = nn.Conv2d(self.input_size * 2, self.input_size * 2, 1, 1)
        assert(self.k <= self.num_experts) 

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating."""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        # 确保所有tensor的形状一致
        normal = Normal(self.mean, self.std)
        
        # 扩展noise_stddev到与clean_values相同的形状
        noise_stddev = noise_stddev.expand_as(clean_values)
        
        # 计算概率时添加数值稳定性
        valid_mask = noise_stddev > 0
        
        # 初始化概率tensor
        prob_if_in = torch.zeros_like(clean_values)
        prob_if_out = torch.zeros_like(clean_values)
        
        if valid_mask.any():
            # 确保所有参与计算的tensor形状一致
            clean_valid = clean_values[valid_mask]
            threshold_in_valid = threshold_if_in.expand_as(clean_values)[valid_mask]
            threshold_out_valid = threshold_if_out.expand_as(clean_values)[valid_mask]
            noise_valid = noise_stddev[valid_mask]
            
            prob_if_in[valid_mask] = normal.cdf(
                (clean_valid - threshold_in_valid) / noise_valid
            )
            prob_if_out[valid_mask] = normal.cdf(
                (clean_valid - threshold_out_valid) / noise_valid
            )
        
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            # 添加clip操作防止数值不稳定
            noise_stddev = torch.clamp(
                self.softplus(raw_noise_stddev) + noise_epsilon,
                min=1e-7,
                max=1.0
            )
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            # 添加数值稳定性检查
            load = self._prob_in_top_k(
                clean_logits, 
                noisy_logits, 
                noise_stddev.clamp(min=1e-7), # 确保除数不为0
                top_logits
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, guide, prompt):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """ 
        
        # import pdb 
        # pdb.set_trace() 
        
        B, C, H, W = x.shape
        prompt = prompt.unsqueeze(-1).unsqueeze(-1).expand_as(x)


        x_tmp = torch.cat((x, guide), dim=1)
        x_tmp = self.conv1(x_tmp)

        x_guide = torch.cat((x_tmp, prompt), dim=1)
        x_guide = self.conv2(x_guide)
        x_tmp = self.conv3(F.gelu(x_guide) * x_tmp)
        x_gating = rearrange(x_tmp, 'b c h w -> (b h w) c')

        # print(C)

        x = rearrange(x, 'b c h w -> (b h w) c')
        # prompt = rearrange(prompt, 'b c h w -> (b h w) c')
        #
        # x_gating = torch.cat((x, prompt), dim=1) #[B, 2C, H, W]
        # print(x_gating.shape)
        
        gates, load = self.noisy_top_k_gating(x_gating, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs) 
        
        y = rearrange(y, '(b h w) c -> b c h w', b=B, h=H, w=W)
        
        return y, loss


class RIN(nn.Module):
    def __init__(self, in_dim, atom_num=16, atom_dim=256):
        super(RIN, self).__init__()
        
        # Condtion network 
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3,3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1)) 
        
        self.lastOut = nn.Linear(32, atom_num) 
        self.act = nn.GELU()
        
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out) 
        logits = F.softmax(out, -1) 
        out = logits @ self.dictionary 
        out = self.act(out)
        return out 

class RIN_Clip(nn.Module):
    def __init__(self, in_dim, atom_num=16, atom_dim=256):
        super(RIN_Clip, self).__init__()
        
        # Condtion network 
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1)) 
        
        self.lastOut = nn.Linear(32, atom_num) 
        self.fuse_layer = nn.Linear(640, 128)
        self.act = nn.GELU()
        
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x, clip_feature):
        B, C, H, W = x.shape
        out = self.CondNet(x)

        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1) # B, C, 1, 1 -> B, C
        out = self.lastOut(out) 
        logits = F.softmax(out, -1) 
        out = logits @ self.dictionary 
        out = self.act(out) # B, 128
        clip_feature = clip_feature.unsqueeze(0).repeat(B,1) # B, 512
        out = self.fuse_layer(torch.cat((out, clip_feature), dim=1)) # B, 128
        return out


class RIN_Clip_test(nn.Module):
    def __init__(self, in_dim_tar, in_dim_guide, atom_num=16, atom_dim=256):
        super(RIN_Clip_test, self).__init__()

        # Condtion network
        hidden_dim = 64
        self.CondNet1 = nn.Sequential(nn.Conv2d(in_dim_tar, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1))

        self.CondNet2 = nn.Sequential(nn.Conv2d(in_dim_guide, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1))

        self.refine_tar = nn.Sequential(nn.Conv2d(64, 32, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(32, 32, 1))
        self.refine_guide = nn.Sequential(nn.Conv2d(64, 32, 1), nn.LeakyReLU(0.1, True),
                                        nn.Conv2d(32, 32, 1))
        self.refine_out = nn.Conv2d(64, 64, 1)

        self.linear1 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(512, 64)

        self.lastOut = nn.Linear(64, atom_num)
        self.fuse_layer = nn.Linear(640, 128)
        self.act = nn.GELU()

        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, tar, guide, clip_feature):
        B, C, H, W = tar.shape
        out_tar = self.CondNet1(tar)
        out_guide = self.CondNet2(guide)
        out_tar1 = torch.cat((out_tar, out_guide), dim=1)
        out_guide1 = torch.cat((out_guide, out_tar), dim=1)
        out_tar = self.refine_tar(out_tar1)
        out_guide = self.refine_guide(out_guide1)
        out = torch.cat((out_tar, out_guide), dim=1)
        out = self.refine_out(out)

        out = (out * self.linear1(clip_feature).view(1, 64, 1, 1) + self.linear2(clip_feature).view(1, 64, 1, 1)) + out

        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)  # B, C, 1, 1 -> B, C
        out = self.lastOut(out)
        logits = F.softmax(out, -1)
        out = logits @ self.dictionary
        out = self.act(out)  # B, 128

        # clip_feature = clip_feature.unsqueeze(0).repeat(B, 1)  # B, 512
        # out = self.fuse_layer(torch.cat((out, clip_feature), dim=1))  # B, 128
        return out
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Channel_Routing(nn.Module):

    def __init__(self, atom_dim, dim):
        super(Channel_Routing, self).__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True) 

    def forward(self, x, prompt):
        gating_factors = torch.sigmoid(self.fc(prompt))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        out = x * self.gamma + self.beta  
        out = out * gating_factors 
             
        return x + out


class Spatial_Routing(nn.Module):
    def __init__(self, atom_dim, dim, ffn_expansion_factor):
        super(Spatial_Routing, self).__init__() 
        
        self.fc = nn.Linear(atom_dim, dim) 
        self.moe = MoE(dim, dim, mlp_ratio=ffn_expansion_factor, num_experts=4, noisy_gating=True, use_experts=2) 

    def forward(self, x, guide, prompt):
        d = self.fc(prompt) 
        out, loss = self.moe(x, guide, d)
        return out + x, loss




##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
class IFM(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim=32):
        super(IFM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.softmax  = nn.Softmax(dim=-1)
        self.sig = nn.Sigmoid()
    def forward(self,x, prior):
        
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)
        # print(attention.size(),x.size())
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x),dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p),dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out


##########################################################################
##---------- MOE -----------------------
class MOE_IFM_Clip(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 42,
        num_blocks = [5,7,7,9], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(MOE_IFM_Clip, self).__init__()

        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 
        
        # atom_dim = 256 
        # atom_num = 32 
        atom_dim = 128
        atom_num = 16 
        # self.dict_generator1 = RIN_Clip(in_dim=4, atom_num=atom_num, atom_dim=atom_dim) # depth
        # self.dict_generator2 = RIN_Clip(in_dim=2, atom_num=atom_num, atom_dim=atom_dim) # mri
        # self.dict_generator3 = RIN_Clip(in_dim=5, atom_num=atom_num, atom_dim=atom_dim) # pan

        self.generator1 = RIN_Clip_test(in_dim_tar=1, in_dim_guide=3, atom_num=atom_num, atom_dim=atom_dim)  # depth
        self.generator2 = RIN_Clip_test(in_dim_tar=1, in_dim_guide=1, atom_num=atom_num, atom_dim=atom_dim)  # mri
        self.generator3 = RIN_Clip_test(in_dim_tar=4, in_dim_guide=1, atom_num=atom_num, atom_dim=atom_dim)  # pan
        
        self.spatial_routing_encoder_level1 = Spatial_Routing(atom_dim = atom_dim, dim=dim,  ffn_expansion_factor=ffn_expansion_factor) 
        self.spatial_routing_encoder_level2 = Spatial_Routing(atom_dim = atom_dim, dim=int(dim*2**1), ffn_expansion_factor=ffn_expansion_factor)
        self.spatial_routing_encoder_level3 = Spatial_Routing(atom_dim = atom_dim, dim=int(dim*2**2), ffn_expansion_factor=ffn_expansion_factor)
        
        # self.channel_routing_latent = Channel_Routing(atom_dim = atom_dim, dim=int(dim*2**3)) 
        # self.channel_routing_decoder_level3 = Channel_Routing(atom_dim = atom_dim, dim=int(dim*2**2)) 
        # self.channel_routing_decoder_level2 = Channel_Routing(atom_dim = atom_dim, dim=int(dim*2**1)) 
        # self.channel_routing_decoder_level1 = Channel_Routing(atom_dim = atom_dim, dim=int(dim*2**1)) 
        
        # self.patch_embed1 = OverlapPatchEmbed(4, dim) # depth
        # self.patch_embed2 = OverlapPatchEmbed(2, dim) # mri
        # self.patch_embed3 = OverlapPatchEmbed(5, dim) # pan
        
        self.inp_tar_embed1 = OverlapPatchEmbed(1, dim) # depth
        self.inp_guide_embed1 = OverlapPatchEmbed(3, dim) # depth
        self.inp_tar_embed2 = OverlapPatchEmbed(1, dim) # mri
        self.inp_guide_embed2 = OverlapPatchEmbed(1, dim) # mri
        self.inp_tar_embed3 = OverlapPatchEmbed(4, dim) # pan
        self.inp_guide_embed3 = OverlapPatchEmbed(1, dim) # pan
        
        self.ifm = IFM(in_dim=dim)
        self.inp_feature_fuse = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2_tar = Downsample(dim) ## From Level 1 to Level 2
        self.down1_2_guide = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3_tar = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.down2_3_guide = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        # self.output1 = nn.Conv2d(int(dim*2**1), 1, kernel_size=3, stride=1, padding=1, bias=bias) # depth
        # self.output2 = nn.Conv2d(int(dim*2**1), 1, kernel_size=3, stride=1, padding=1, bias=bias) # mri
        # self.output3 = nn.Conv2d(int(dim*2**1), 4, kernel_size=3, stride=1, padding=1, bias=bias) # pan
        self.output1 = nn.Conv2d(int(dim*2**1), 1, kernel_size=1, bias=bias) # depth
        self.output2 = nn.Conv2d(int(dim*2**1), 1, kernel_size=1, bias=bias) # mri
        self.output3 = nn.Conv2d(int(dim*2**1), 4, kernel_size=1, bias=bias) # pan

    def forward(self, inp_tar, inp_guide, clip_feature): 
        _,C1,_,_ = inp_tar.shape
        _,C2,H,W = inp_guide.shape
        inp_tar = F.interpolate(inp_tar, size = (H,W), mode = 'bilinear')

        # inp_img = torch.concat([inp_tar,inp_guide],dim=1)

        if C1 + C2 == 4: # depth
            inp_guide_level1 = self.inp_guide_embed1(inp_guide)
            inp_tar_feature, inp_guide_feature = self.ifm(self.inp_tar_embed1(inp_tar), inp_guide_level1)
            inp_enc_level1 = self.inp_feature_fuse(torch.cat([inp_tar_feature, inp_guide_feature], dim=1))
            # prompt = self.dict_generator1(inp_img, clip_feature)
            prompt = self.generator1(inp_tar, inp_guide, clip_feature)

        elif C1 + C2 == 2 : # mri
            inp_guide_level1 = self.inp_guide_embed2(inp_guide)
            inp_tar_feature, inp_guide_feature = self.ifm(self.inp_tar_embed2(inp_tar), inp_guide_level1)
            inp_enc_level1 = self.inp_feature_fuse(torch.cat([inp_tar_feature, inp_guide_feature], dim=1))
            # prompt = self.dict_generator2(inp_img, clip_feature)
            prompt = self.generator2(inp_tar, inp_guide, clip_feature)

        elif C1 + C2 == 5: # pan
            inp_guide_level1 = self.inp_guide_embed3(inp_guide)
            inp_tar_feature, inp_guide_feature = self.ifm(self.inp_tar_embed3(inp_tar), inp_guide_level1)
            inp_enc_level1 = self.inp_feature_fuse(torch.cat([inp_tar_feature, inp_guide_feature], dim=1))
            # prompt = self.dict_generator3(inp_img, clip_feature)
            prompt = self.generator3(inp_tar, inp_guide, clip_feature)

        inp_enc_level1, loss_tmp = self.spatial_routing_encoder_level1(inp_enc_level1, inp_guide_level1, prompt)
        loss_importance = loss_tmp
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 

        inp_enc_level2 = self.down1_2_tar(out_enc_level1)
        inp_guide_level2 = self.down1_2_guide(inp_guide_level1)
        inp_enc_level2, loss_tmp = self.spatial_routing_encoder_level2(inp_enc_level2, inp_guide_level2, prompt)
        loss_importance = loss_importance + loss_tmp
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 

        inp_enc_level3 = self.down2_3_tar(out_enc_level2)
        inp_guide_level3 = self.down2_3_guide(inp_guide_level2)

        inp_enc_level3, loss_tmp = self.spatial_routing_encoder_level3(inp_enc_level3, inp_guide_level3, prompt)
        loss_importance = loss_importance + loss_tmp
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3)
        # inp_enc_level4 = self.channel_routing_latent(inp_enc_level4, prompt)
        latent = self.latent(inp_enc_level4) 
        
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        # inp_dec_level3 = self.channel_routing_decoder_level3(inp_dec_level3, prompt)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        # inp_dec_level2 = self.channel_routing_decoder_level2(inp_dec_level2, prompt)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1) 
        # inp_dec_level1 = self.channel_routing_decoder_level1(inp_dec_level1, prompt)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        
        out_dec_level1 = self.refinement(out_dec_level1) 

        # out_dec_level1 = self.output(out_dec_level1) + inp_img
        if C1+C2 == 4:  # depth
            out_dec_level1 = self.output1(out_dec_level1) + inp_tar
        elif C1+C2 ==2: # mri
            out_dec_level1 = self.output2(out_dec_level1) + inp_tar
        else:           # pan
            out_dec_level1 = self.output3(out_dec_level1) + inp_tar


        if self.training:
            return out_dec_level1, loss_importance 
        else: 
            return out_dec_level1 

if __name__ == '__main__':
    depth_1, depth_2 = torch.rand(1, 1, 64, 64), torch.rand(1, 3, 256, 256)
    mri_1, mri_2 = torch.rand(1, 1, 60, 60), torch.rand(1, 1, 240, 240)
    pan_1, pan_2 = torch.rand(1, 4, 32, 32), torch.rand(1, 1, 128, 128)
    # clip_feature = torch.rand(512)
    # clip_feature = clip_feature.type(torch.FloatTensor).cuda()
    depth_1, depth_2 = depth_1.type(torch.FloatTensor).cuda(), depth_2.type(torch.FloatTensor).cuda()
    mri_1, mri_2 = mri_1.type(torch.FloatTensor).cuda(), mri_2.type(torch.FloatTensor).cuda()
    pan_1, pan_2 = pan_1.type(torch.FloatTensor).cuda(), pan_2.type(torch.FloatTensor).cuda()
    
    model = MOE_IFM_Clip(dim=22, num_blocks=[3, 4, 4, 5])
    # model.cuda()
    # model.eval()

    # clip_feature = clip_feature.type(torch.FloatTensor)
    # depth_1, depth_2 = depth_1.type(torch.FloatTensor), depth_2.type(torch.FloatTensor)
    # mri_1, mri_2 = mri_1.type(torch.FloatTensor), mri_2.type(torch.FloatTensor)
    # pan_1, pan_2 = pan_1.type(torch.FloatTensor), pan_2.type(torch.FloatTensor)
    # model = MOE_IFM_Clip(dim=22, num_blocks=[3, 4, 4, 5])
    # model.eval()

    # out_depth = model(depth_1, depth_2, clip_feature)
    # out_mri = model(mri_1,mri_2, clip_feature)
    # out_pan = model(pan_1, pan_2, clip_feature)
    print(sum(p.numel() for p in model.parameters() )/1e6)
    # print(out_depth.shape, out_mri.shape, out_pan.shape)

    