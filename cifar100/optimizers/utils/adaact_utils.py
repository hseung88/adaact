import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaActStats:
    @staticmethod
    def update_linear(module, actv):
        actv = actv.view(-1, actv.size(-1))
        # augment the activations if the layer has a bias term
        if module.bias is not None:
            actv = torch.cat([actv, actv.new_ones((actv.size(0), 1))], 1)
        
        A = torch.mean(actv.pow(2), axis=0)
        return A

    @staticmethod
    def update_conv(module, actv):
        a = extract_patches(actv, module.kernel_size, module.stride, module.padding)
        a = a.view(-1, a.size(-1))

        if module.bias is not None:
            a = torch.cat([a, a.new_ones((a.size(0), 1))], 1)

        A = torch.mean(a.pow(2), axis=0)
        return A

    STAT_UPDATE_FUNC = {
        nn.Linear: update_linear.__func__,
        nn.Conv2d: update_conv.__func__
    }

    @classmethod
    def __call__(cls, module, actv):
        return cls.STAT_UPDATE_FUNC[type(module)](module, actv)


def build_layer_map(model, fwd_hook_fn=None, bwd_hook_fn=None,
                    supported_layers=(nn.Linear, nn.Conv2d)):
    layer_map = {}

    for layer, prefix, params in grad_layers(model):
        if isinstance(layer, supported_layers):
            h_fwd_hook = layer.register_forward_hook(fwd_hook_fn) if fwd_hook_fn else None
            h_bwd_hook = layer.register_full_backward_hook(bwd_hook_fn) if bwd_hook_fn else None
        else:
            h_fwd_hook = None
            h_bwd_hook = None

        layer_map[layer] = {
            'name': prefix,
            'params': params,  # list of tuples; each tuple is of form: (pname, parameter)
            'fwd_hook': h_fwd_hook,
            'bwd_hook': h_bwd_hook
        }
    return layer_map


def moving_average(new_val, stat, decay):
    stat.mul_(decay).add_(new_val, alpha=1.0 - decay)


def extract_patches(x, kernel_size, stride, padding):
    """
    x: input feature map of shape (B x C x H x W)
    kernel_size: the kernel size of the conv filter (tuple of two elements)
    stride: the stride of conv operation  (tuple of two elements)
    padding: number of paddings. be a tuple of two elements

    return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims

    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def reshape_grad(layer):
    """
    returns the gradient reshaped for KFAC, shape=[batch_size, output_dim, input_dim]
    """
    classname = layer.__class__.__name__

    g = layer.weight.grad

    if classname == 'Conv2d':
        grad_mat = g.view(g.size(0), -1)  # n_filters * (in_c * kw * kh)
    else:
        grad_mat = g

    # include the bias into the weight
    if layer.bias is not None:
        grad_mat = torch.cat([grad_mat, layer.bias.grad.view(-1, 1)], 1)

    return grad_mat

def grad_layers(module, memo=None, prefix=''):
    if memo is None:
        memo = set()

    if module not in memo:
        memo.add(module)

        if bool(module._modules):
            for name, module in module._modules.items():
                if module is None:
                    continue
                sub_prefix = prefix + ('.' if prefix else '') + name
                for ll in grad_layers(module, memo, sub_prefix):
                    yield ll
        else:
            if bool(module._parameters):
                grad_param = []

                for pname, param in module._parameters.items():
                    if param is None:
                        continue

                    if param.requires_grad:
                        grad_param.append((pname, param))

                if grad_param:
                    yield module, prefix, grad_param