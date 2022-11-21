import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepPoly:
    def __init__(self, size, lb, ub):
        self.slb = torch.cat([torch.diag(torch.ones(size)), torch.zeros(size).unsqueeze(1)], dim=1)
        self.sub = self.slb
        self.lb = lb
        self.ub = ub
        self.history = []
        self.layers = 0
        self.is_relu = False

    def save(self):
        """ Save all constraints for the back substitution """
        lb = torch.cat([self.lb, torch.ones(1)])
        ub = torch.cat([self.ub, torch.ones(1)])
        if self.is_relu:
            # relu layer
            slb = self.slb
            sub = self.sub
        else:
            # other layers
            keep_bias = torch.zeros(1, self.slb.shape[1])
            keep_bias[0, self.slb.shape[1] - 1] = 1
            slb = torch.cat([self.slb, keep_bias], dim=0)
            sub = torch.cat([self.sub, keep_bias], dim=0)
        # layer num
        self.layers += 1
        # record each layer
        self.history.append((slb, sub, lb, ub, self.is_relu))
        return self

    def compute_verify_result(self, true_label):
        self.save()
        n = self.slb.shape[0] - 1
        unit = torch.diag(torch.ones(n))
        weights = torch.cat((-unit[:, :true_label], torch.ones(n, 1), -unit[:, true_label:], torch.zeros(n, 1)), dim=1)

        for i in range(self.layers, 0, -1):
            weights = self.resolve(weights, i - 1, lower=True)

        return weights

################################################################## to understand
    def resolve(self, constrains, layer, lower=True):
        """
        lower = True: return the lower bound
        lower = False: return the upper bound
        """
        # distinguish the sign of the coefficients of the constraints
        pos_coeff = F.relu(constrains)
        neg_coeff = F.relu(-constrains)
        layer_info = self.history[layer]
        is_relu = layer_info[-1]
        if layer == 0:
            # layer_info[2],layer_info[3]: concrete lower and upper bound
            lb, ub = layer_info[2], layer_info[3]
        else:
            # layer_info[0],layer_info[1]: symbolic lower and upper bound
            lb, ub = layer_info[0], layer_info[1]
        if not lower:
            lb, ub = ub, lb
        if is_relu:
            lb_diag, lb_bias = lb[0], lb[1]
            ub_diag, ub_bias = ub[0], ub[1]
            lb_bias = torch.cat([lb_bias, torch.ones(1)])
            ub_bias = torch.cat([ub_bias, torch.ones(1)])

            m1 = torch.cat([pos_coeff[:, :-1] * lb_diag, torch.matmul(pos_coeff, lb_bias).unsqueeze(1)], dim=1)
            m2 = torch.cat([neg_coeff[:, :-1] * ub_diag, torch.matmul(neg_coeff, ub_bias).unsqueeze(1)], dim=1)
            return m1 - m2
        else:
            return torch.matmul(pos_coeff, lb) - torch.matmul(neg_coeff, ub)



class DPReLU(nn.Module):
    def __init__(self, in_features):
        super(DPReLU, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.alpha = torch.nn.Parameter(torch.ones(in_features) * 0.3)
        self.alpha.requires_grad = True

    def forward(self, x):
        x.save()
        low, up = x.lb, x.ub
        mask_1, mask_2 = low.ge(0), up.le(0)
        mask_3 = ~(mask_1 | mask_2)

        '''
        low > 0
        '''
        slope_low_1 = (F.relu(up) - F.relu(low)) / (up - low)
        bias_low_1 = F.relu(low) - slope_low_1 * low
        slope_up_1 = (F.relu(up) - F.relu(low)) / (up - low)
        bias_up_1 = F.relu(up) - slope_up_1 * up
        '''
        up < 0
        '''
        slope_low_2 = (F.relu(up) - F.relu(low)) / (up - low)
        bias_low_2 = F.relu(low) - slope_low_2 * low
        slope_up_2 = F.relu(up) - F.relu(low) / (up - low)
        bias_up_2 = F.relu(up) - slope_up_2 * up
        '''
        low < 0 < up
        '''
        # print("ALPHA: ", self.alpha)
        slope_low_3 = torch.tan(self.alpha)
        bias_low_3 = slope_low_3 * low - slope_low_3 * low
        slope_up_3 = (F.relu(up) - F.relu(low)) / (up - low)
        bias_up_3 = F.relu(up) - slope_up_3 * up

        curr_slb = slope_low_1 * mask_1 + slope_low_2 * mask_2 + slope_low_3 * mask_3
        curr_slb_bias = bias_low_1 * mask_1 + bias_low_2 * mask_2 + bias_low_3 * mask_3
        curr_sub = slope_up_1 * mask_1 + slope_up_2 * mask_2 + slope_up_3 * mask_3
        curr_sub_bias = bias_up_1 * mask_1 + bias_up_2 * mask_2 + bias_up_3 * mask_3

        x.lb = F.relu(low)
        x.ub = F.relu(up)
        x.slb = torch.cat([curr_slb.unsqueeze(0), curr_slb_bias.unsqueeze(0)], dim=0)
        x.sub = torch.cat([curr_sub.unsqueeze(0), curr_sub_bias.unsqueeze(0)], dim=0)
        x.is_relu = True
        return x


class DPLinear(nn.Module):
    def __init__(self, nested: nn.Linear):
        super(DPLinear, self).__init__()
        self.weight = nested.weight.detach()
        self.bias = nested.bias.detach()
        self.in_features = nested.in_features
        self.out_features = nested.out_features

    def forward(self, x):
        x.save()
        # append bias as last column
        init_slb = torch.cat([self.weight, self.bias.unsqueeze(1)], dim=1)
        x.lb = init_slb
        x.ub = init_slb
        x.slb = init_slb
        x.sub = init_slb
        for i in range(x.layers, 0, -1):
            x.lb = x.resolve(x.lb, i - 1, lower=True)
            x.ub = x.resolve(x.ub, i - 1, lower=False)
        x.is_relu = False
        return x


class DPConv(nn.Module):
    def  __init__(self, nested: nn.Conv2d):
        super(DPConv, self).__init__()
