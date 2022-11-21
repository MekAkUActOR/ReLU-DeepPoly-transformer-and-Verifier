import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepPoly:
    def __init__(self, size, low_bound, up_bound):
        self.lowbound = low_bound
        self.upbound = up_bound
        self.sym_lowbound = torch.cat([torch.diag(torch.ones(size)), torch.zeros(size).unsqueeze(1)], dim=1)
        self.sym_upbound = self.sym_lowbound
        self.history = []
        self.layers = 0
        self.is_relu = False

    def save(self):
        """ Save constraints for the backsubstitution """
        low_bound = torch.cat([self.lowbound, torch.ones(1)])
        up_bound = torch.cat([self.upbound, torch.ones(1)])
        if self.is_relu:
            sym_lowbound = self.sym_lowbound
            sym_upbound = self.sym_upbound
        else:
            # other layers
            keep_bias = torch.zeros(1, self.sym_lowbound.shape[1])
            keep_bias[0, self.sym_lowbound.shape[1] - 1] = 1
            sym_lowbound = torch.cat([self.sym_lowbound, keep_bias], dim=0)
            sym_upbound = torch.cat([self.sym_upbound, keep_bias], dim=0)
            # layer num
        self.layers += 1
        # record each layer
        self.history.append((sym_lowbound, sym_upbound, low_bound, up_bound, self.is_relu))
        return self

    def compute_verify_result(self, true_label):
        self.save()
        n = self.sym_lowbound.shape[0] - 1 # why
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
            low_bound, up_bound = layer_info[2], layer_info[3]
        else:
            # layer_info[0],layer_info[1]: symbolic lower and upper bound
            low_bound, up_bound = layer_info[0], layer_info[1]
        if not lower:
            low_bound, up_bound = up_bound, low_bound
        if is_relu:
            low_diag, low_bias = low_bound[0], low_bound[1]
            up_diag, up_bias = up_bound[0], up_bound[1]
            low_bias = torch.cat([low_bias, torch.ones(1)])
            up_bias = torch.cat([up_bias, torch.ones(1)])

            m1 = torch.cat([pos_coeff[:, :-1] * low_diag, torch.matmul(pos_coeff, low_bias).unsqueeze(1)], dim=1)
            m2 = torch.cat([neg_coeff[:, :-1] * up_diag, torch.matmul(neg_coeff, up_bias).unsqueeze(1)], dim=1)
            return m1 - m2
        else:
            return torch.matmul(pos_coeff, low_bound) - torch.matmul(neg_coeff, up_bound)



class DPReLU(nn.Module):
    def __init__(self, size):
        super(DPReLU, self).__init__()
        self.in_features = size
        self.out_features = size
        self.alpha = torch.nn.Parameter(torch.ones(size))

    def forward(self, x):
        x.save()
        low, up = x.lowbound, x.upbound
        mask_1, mask_2 = low.ge(0), up.le(0)
        mask_3 = ~(mask_1 | mask_2)
        print("DPReLU: ", low.shape, up.shape, self.alpha.shape)

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
        slope_low_3 = self.alpha
        bias_low_3 = self.alpha * low - self.alpha * low
        slope_up_3 = (F.relu(up) - F.relu(low)) / (up - low)
        bias_up_3 = F.relu(up) - slope_up_3 * up

        curr_slb = slope_low_1 * mask_1 + slope_low_2 * mask_2 + slope_low_3 * mask_3
        curr_slb_bias = bias_low_1 * mask_1 + bias_low_2 * mask_2 + bias_low_3 * mask_3
        curr_sub = slope_up_1 * mask_1 + slope_up_2 * mask_2 + slope_up_3 * mask_3
        curr_sub_bias = bias_up_1 * mask_1 + bias_up_2 * mask_2 + bias_up_3 * mask_3

        x.lowbound = F.relu(low)
        x.upbound = F.relu(up)
        x.sym_lowbound = torch.cat([curr_slb.unsqueeze(0), curr_slb_bias.unsqueeze(0)], dim=0)
        x.sym_upbound = torch.cat([curr_sub.unsqueeze(0), curr_sub_bias.unsqueeze(0)], dim=0)
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


