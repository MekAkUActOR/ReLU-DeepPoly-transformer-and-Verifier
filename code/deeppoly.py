import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import resnet

class DeepPoly:
    def __init__(self, size, lb, ub):
        self.slb = torch.cat([torch.diag(torch.ones(size)), torch.zeros(size).unsqueeze(1)], dim=1)
        self.sub = self.slb
        self.lb = lb
        self.ub = ub

        self.history = []
        self.layers = 0

    def save(self):
        # print("save ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # """ Save all constraints for the back substitution """
        lb = torch.cat([self.lb, torch.ones(1)])
        # print("save lb,", lb.shape)
        ub = torch.cat([self.ub, torch.ones(1)])
        # print("save ub,", ub.shape)
        keep_bias = torch.zeros(1, self.slb.shape[1])
        keep_bias[0, self.slb.shape[1] - 1] = 1
        slb = torch.cat([self.slb, keep_bias], dim=0)
        sub = torch.cat([self.sub, keep_bias], dim=0)

        # layer num
        self.layers += 1
        # record each layer
        self.history.append((slb, sub, lb, ub))
        return self

    def compute_verify_result(self, true_label):
        self.save()
        n = self.slb.shape[0] - 1
        unit = torch.diag(torch.ones(n))
        weights = torch.cat((-unit[:, :true_label], torch.ones(n, 1), -unit[:, true_label:], torch.zeros(n, 1)), dim=1)
        # print("compute_verify_result,", weights.shape)
        # print("===========================================================================")

        for i in range(self.layers, 0, -1):
            weights = self.resolve(weights, i - 1, lower=True)
        # print("----------------------------------------------------------")
        # print("compute_verify_result,", weights.shape, weights)

        return weights

    # TODO: implement Conv in resolve
    def resolve(self, constrains, layer, lower=True):
        # print("resolve >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        lower = True: return the lower bound
        lower = False: return the upper bound
        """
        # distinguish the sign of the coefficients of the constraints
        pos_coeff = F.relu(constrains)
        neg_coeff = -F.relu(-constrains)
        layer_info = self.history[layer]
        if layer == 0:
            # layer_info[2],layer_info[3]: concrete lower and upper bound
            lb, ub = layer_info[2], layer_info[3]
        else:
            # layer_info[0],layer_info[1]: symbolic lower and upper bound
            lb, ub = layer_info[0], layer_info[1]
        if not lower:
            lb, ub = ub, lb
        m1 = torch.matmul(pos_coeff, lb)
        m2 = torch.matmul(neg_coeff, ub)
        return m1 + m2


class DPReLU(nn.Module):
    def __init__(self, in_features):
        super(DPReLU, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        # self.alpha = torch.nn.Parameter(torch.ones(in_features) * 0.5)
        # self.alpha = torch.nn.Parameter(torch.rand(in_features) * 0.7854)
        self.alpha = torch.nn.Parameter(torch.rand(in_features))
        # self.alpha = torch.nn.Parameter(torch.normal(mean=0.5, std=torch.ones(in_features)*0.5))
        # self.alpha.data.clamp_(0, 1)
        self.alpha.requires_grad = True

    def forward(self, x):
        x.save()
        low, up = x.lb, x.ub
        mask_1, mask_2 = low.ge(0), up.le(0)
        mask_3 = ~(mask_1 | mask_2)

        '''
        low > 0
        '''
        # slope_low_1 = (F.relu(up) - F.relu(low)) / (up - low)
        # bias_low_1 = F.relu(low) - slope_low_1 * low
        # slope_up_1 = (F.relu(up) - F.relu(low)) / (up - low)
        # bias_up_1 = F.relu(up) - slope_up_1 * up
        slope_low_1 = torch.ones(self.in_features)
        bias_low_1 = torch.zeros(self.in_features)
        slope_up_1 = torch.ones(self.in_features)
        bias_up_1 = torch.zeros(self.in_features)
        '''
        up < 0
        '''
        # slope_low_2 = (F.relu(up) - F.relu(low)) / (up - low)
        # bias_low_2 = F.relu(low) - slope_low_2 * low
        # slope_up_2 = F.relu(up) - F.relu(low) / (up - low)
        # bias_up_2 = F.relu(up) - slope_up_2 * up
        slope_low_2 = torch.zeros(self.in_features)
        bias_low_2 = torch.zeros(self.in_features)
        slope_up_2 = torch.zeros(self.in_features)
        bias_up_2 = torch.zeros(self.in_features)
        '''
        low < 0 < up
        '''
        # print("ALPHA: ", self.alpha)
        # slope_low_3 = torch.tan(self.alpha)
        # slope_low_3 = self.alpha
        # bias_low_3 = slope_low_3 * low - slope_low_3 * low
        # slope_up_3 = (F.relu(up) - F.relu(low)) / (up - low)
        # bias_up_3 = F.relu(up) - slope_up_3 * up
        slope_low_3 = self.alpha
        bias_low_3 = slope_low_3 * low - slope_low_3 * low
        slope_up_3 = (F.relu(up) - F.relu(low)) / (up - low)
        bias_up_3 = F.relu(up) - slope_up_3 * up

        curr_slb = slope_low_1 * mask_1 + slope_low_2 * mask_2 + slope_low_3 * mask_3
        curr_slb_bias = bias_low_1 * mask_1 + bias_low_2 * mask_2 + bias_low_3 * mask_3
        curr_sub = slope_up_1 * mask_1 + slope_up_2 * mask_2 + slope_up_3 * mask_3
        curr_sub_bias = bias_up_1 * mask_1 + bias_up_2 * mask_2 + bias_up_3 * mask_3

        x.lb = F.relu(low)
        x.ub = F.relu(up)
        x.slb = torch.cat([torch.diag(curr_slb), curr_slb_bias.unsqueeze(1)], dim=1)
        x.sub = torch.cat([torch.diag(curr_sub), curr_sub_bias.unsqueeze(1)], dim=1)
        # x.is_relu = True
        return x


class DPFlatten(nn.Module):
    def __init__(self):
        super(DPFlatten, self).__init__()
        self.kernel_size = 1
        self.stride = 1
        self.padding = 0

    def forward(self, x):
        x.save()
        x.lb = torch.flatten(x.lb, start_dim=1).unsqueeze(2).unsqueeze(3)
        x.ub = torch.flatten(x.ub, start_dim=1).unsqueeze(2).unsqueeze(3)
        x.slb = torch.cat([torch.diag(torch.ones(x.lb.shape[1])), torch.zeros(x.lb.shape[1]).unsqueeze(1)], dim=1).unsqueeze(2).unsqueeze(3)
        x.sub = x.slb
        x.kernel_size = self.kernel_size
        x.stride = self.stride
        x.padding = self.padding
        # x.is_relu = False
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
        # print("Linear weight", self.weight.shape)
        # print("Linear bias", self.bias.shape)
        init_slb = torch.cat([self.weight, self.bias.unsqueeze(1)], dim=1)
        # print("Linear init_slb", init_slb.shape)
        x.lb = init_slb
        x.ub = init_slb
        x.slb = init_slb
        x.sub = init_slb
        for i in range(x.layers, 0, -1):
            x.lb = x.resolve(x.lb, i - 1, lower=True)
            x.ub = x.resolve(x.ub, i - 1, lower=False)
        # x.is_relu = False
        return x


class DPConv(nn.Module):
    def __init__(self, nested: nn.Conv2d, in_feature):
        super(DPConv, self).__init__()
        self.weight = nested.weight.detach()
        self.bias = nested.bias.detach() # tensor[out_channels]
        self.in_channels = nested.in_channels
        self.out_channels = nested.out_channels
        self.kernel_size = nested.kernel_size
        self.stride = nested.stride
        self.padding = nested.padding
        # self.padding_mode = nested.padding_mode
        # self.dilation = nested.dilation
        # self.groups = nested.groups
        # print("DPConv create", in_feature)
        # print("DPConv create", self.in_channels)
        # print("DPConv create", self.out_channels)
        img_height = math.floor(math.sqrt(in_feature / self.in_channels))
        self.in_features = (
            self.in_channels,
            img_height,
            img_height,
        )
        # print("DPConv create", self.in_features)
        # print("DPConv create", self.kernel_size)
        # print("DPConv create", self.stride)
        # print("DPConv create", self.padding)
        self.out_features = self.out_channels * \
                           math.floor((self.in_features[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1) * \
                           math.floor((self.in_features[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
        self.weightMtrx = self.get_sparse_conv_weights_matrix(self.weight, self.in_features, self.bias, self.stride[0], self.padding[0])

    def forward(self, x):
        x.save()
        init_slb = self.weightMtrx
        # print("Conv init_slb", init_slb.shape, init_slb)
        x.lb = init_slb
        x.ub = init_slb
        x.slb = init_slb
        x.sub = init_slb
        for i in range(x.layers, 0, -1):
            x.lb = x.resolve(x.lb, i - 1, lower=True)
            x.ub = x.resolve(x.ub, i - 1, lower=False)
        # print("Conv init_slb x.lb", x.lb)
        # print("Conv init_slb x.ub", x.ub)
        # x.is_relu = False
        return x

    @staticmethod
    def get_sparse_conv_weights_matrix(conv_weights, in_feature, bias, stride=1, padding=0):
        """
        conv_weights: [(c_out, c_in, kernel_size, kernel_size)]
        # inputs: [(c_in, input_height, input_width)]
        inputs: [(pixels: c_in * input_height * input_width)]
        bias: [(c_out)]
        """
        c_out = conv_weights.shape[0]
        c_in = conv_weights.shape[1]
        kernel_size = conv_weights.shape[2]
        input_height = in_feature[1]
        input_width = in_feature[2]
        temp_input = torch.ones(input_height, input_width)
        matrix_mask = torch.flatten(pad_image(temp_input, padding))
        output_height = math.floor((input_height - kernel_size + 2 * padding) / stride + 1)
        output_width = math.floor((input_width - kernel_size + 2 * padding) / stride + 1)
        input_pad_h = input_height + 2 * padding
        input_pad_w = input_width + 2 * padding
        matrix_lst = []
        for c, line in enumerate(conv_weights):
            matrix_line_lst = []
            for ker in line:
                kernel_matrix = unroll_kernel_matrix(ker,
                                                     output_height,
                                                     output_width,
                                                     input_pad_h,
                                                     input_pad_w,
                                                     stride)
                # a test
                # kernel_matrix = torch.randn((output_height * output_width, input_pad_h * input_pad_w))
                dense_matrix = dense_weights(kernel_matrix, matrix_mask)
                matrix_line_lst.append(dense_matrix)
            line_matrix = torch.cat(matrix_line_lst, dim=1)
            append_bias = torch.ones(output_height * output_width).unsqueeze(1) * bias[c]
            line_matrix = torch.cat([line_matrix, append_bias], dim=1)
            matrix_lst.append(line_matrix)
        full_matrix = torch.cat(matrix_lst, dim=0)
        return full_matrix


def unroll_kernel_matrix(kernel, output_height, output_width, input_height, input_width, stride=1):
    kernel_matrix = torch.zeros((output_height * output_width, input_height * input_width))
    for i in range(output_height):
        for j in range(output_width):
            for ii in range(kernel.shape[-1]):
                for jj in range(kernel.shape[-1]):
                    kernel_matrix[i * output_width + j, i * input_width * stride + stride * j + ii * input_width + jj] = kernel[ii, jj]
    return kernel_matrix


def dense_weights(matrix, matrix_mask):
    dense_matrix = torch.zeros(matrix.shape[0], int(torch.sum(matrix_mask)))
    k = 0
    for i, m in enumerate(matrix_mask):
        if int(m) == 1:
            dense_matrix[:, k] = matrix[:, i]
            k = k + 1
    return dense_matrix


def pad_image(temp_input, padding):
    pad_zeros = torch.zeros(padding, temp_input.shape[1])
    temp_input = torch.cat([temp_input, pad_zeros], dim=0)
    temp_input = torch.cat([pad_zeros, temp_input], dim=0)
    pad_zeros = torch.zeros(temp_input.shape[0], padding)
    temp_input = torch.cat([temp_input, pad_zeros], dim=1)
    temp_input = torch.cat([pad_zeros, temp_input], dim=1)
    return temp_input


# class DPBatchNorm2d(nn.Module):
#     def __init__(self, in_features):
#         super(DPBatchNorm2d, self).__init__()
#         self.in_features = in_features
#         self.out_features = in_features
#
#     def forward(self, x):
#         x.save()
#         x.lb = self.batchNorm(x.lb)
#         x.ub = self.batchNorm(x.ub)
#         curr_slb =
#         x.slb = torch.cat([curr_slb.unsqueeze(0), curr_slb_bias.unsqueeze(0)], dim=0)
#         x.sub = torch.cat([curr_sub.unsqueeze(0), curr_sub_bias.unsqueeze(0)], dim=0)
#         x.is_relu = True
#         return x
#
#     @staticmethod
#     def batchNorm(x):
#         mean = torch.mean(x)
#         std = torch.std(x, unbiased=False)
#         one_channel_batchnorm = (x - mean) / std
#         return one_channel_batchnorm

# if __name__ == "__main__":
#     test = get_network('cpu', 'net4')
#     print(test)
#     # Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     # Flatten(start_dim=1, end_dim=-1)
#     a = DPConv2d(torch.nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), (1, 28, 28))
#     # print(a.weight)
#     print(a.bias)
#     print(type(a.bias))
#     print(a.bias.shape)
#
#     print(a.weight.shape)
#     b = torch.cat([a.weight, a.bias.unsqueeze(3)], dim=0)
#     print(b.shape)

class DPBasicBlock(nn.Module):
    def __init__(self, nested: resnet.BasicBlock, in_feature):
        super(DPBasicBlock, self).__init__()
        self.in_planes = nested.in_planes
        self.planes = nested.planes
        self.stride = nested.stride
        self.bn = nested.bn
        self.kernel = nested.kernel
        self.expansion = nested.expansion
        self.path_a = nested.path_a
        self.path_b = nested.path_b
        # self.paths = []
        # for modu in nested.modules():
        #     if type(modu) == nn.Sequential:
        #         self.paths.append(modu)
        self.in_feature = in_feature
        img_height = math.floor(math.sqrt(in_feature / self.in_planes))
        self.in_features = (
            self.in_planes,
            img_height,
            img_height,
        )
        temp_img = torch.zeros(self.in_features)
        output_shape = self.path_a(temp_img).shape
        self.out_features = output_shape[0] * output_shape[1] * output_shape[2]

    def forward(self, x):
        lbs = []
        ubs = []
        slbs = []
        subs = []
        for path in self.paths:
            in_feature = self.in_feature
            slb = 1
            sub = 1
            for layer in path:
                if type(layer) == nn.Conv2d:
                    dp_conv = DPConv(layer, in_feature)
                    dp_conv(x)
                    in_feature = dp_conv.out_features
                elif type(layer) == nn.ReLU:
                    dp_relu = DPReLU(in_feature)
                    dp_relu(x)
                    in_feature = dp_relu.out_features
            lbs.append(x.lb)
            ubs.append(x.ub)

        return x

