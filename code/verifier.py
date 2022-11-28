import argparse
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from networks import get_network, get_net_name, NormalizedResnet, FullyConnected, Conv, Normalization
from resnet import ResNet, BasicBlock
from deeppoly import DeepPoly, DPReLU, DPLinear, DPConv, DPBasicBlock


DEVICE = 'cpu'
DTYPE = torch.float32
LR = 0.05
num_iter = 25


# class DeepPoly:
#     def __init__(self, lb, ub, lexpr, uexpr) -> None:
#         """
#         Arguments
#         ---------
#         lb : concrete lower bound
#         ub: concrete upper bound
#         lexpr: symbolic lower bound
#         uexpr: symbolic upper bound
#         """
#         self.lb = lb
#         self.ub = ub
#         self.lexpr = lexpr
#         self.uexpr = uexpr
#         # TODO: remove asserts to improve speed
#         assert not torch.isnan(self.lb).any()
#         assert not torch.isnan(self.ub).any()
#         assert lexpr is None or (
#             (not torch.isnan(self.lexpr[0]).any())
#             and (not torch.isnan(self.lexpr[1]).any())
#         )
#         assert uexpr is None or (
#             (not torch.isnan(self.uexpr[0]).any())
#             and (not torch.isnan(self.uexpr[1]).any())
#         )
#         # TODO: what is dim here
#         self.dim = lb.size()[0]
#
#     @staticmethod
#     def deeppoly_from_perturbation(x, eps):
#         assert eps >= 0, "epsilon must not be negative value"
#         # We only have to verify images with pixel intensities between 0 and 1.
#         # e.g. if perturbation is 0.2 and pixel value is 0.9
#         # we verify range of [0.7， 1.0], instead of [0.7, 1.1]
#         lb = x - eps
#         ub = x + eps
#         lb[lb < 0] = 0
#         ub[ub > 1] = 1
#         return DeepPoly(lb, ub, None, None)
#
#
#
# class DPBackSubstitution:
#     def __init__(self) -> None:
#         pass
#
#     def _get_lb(self, expr_w, expr_b):
#         if len(self.output_dp.lexpr[0].size()) == 2:
#             res_w = (
#                 positive_only(expr_w) @ self.output_dp.lexpr[0]
#                 + negative_only(expr_w) @ self.output_dp.uexpr[0]
#             )
#         else:
#             res_w = (
#                 positive_only(expr_w) * self.output_dp.lexpr[0]
#                 + negative_only(expr_w) * self.output_dp.uexpr[0]
#             )
#         res_b = (
#             positive_only(expr_w) @ self.output_dp.lexpr[1]
#             + negative_only(expr_w) @ self.output_dp.uexpr[1]
#             + expr_b
#         )
#
#         if self.prev_layer == None:
#             return (
#                 positive_only(res_w) @ self.input_dp.lb
#                 + negative_only(res_w) @ self.input_dp.ub
#                 + res_b
#             )
#         else:
#             return self.prev_layer._get_lb(res_w, res_b)
#
#     def _get_ub(self, expr_w, expr_b):
#         if len(self.output_dp.lexpr[0].size()) == 2:
#             res_w = (
#                 positive_only(expr_w) @ self.output_dp.uexpr[0]
#                 + negative_only(expr_w) @ self.output_dp.lexpr[0]
#             )
#         else:
#             res_w = (
#                 positive_only(expr_w) * self.output_dp.uexpr[0]
#                 + negative_only(expr_w) * self.output_dp.lexpr[0]
#             )
#         res_b = (
#             positive_only(expr_w) @ self.output_dp.uexpr[1]
#             + negative_only(expr_w) @ self.output_dp.lexpr[1]
#             + expr_b
#         )
#
#         if self.prev_layer == None:
#             return (
#                 positive_only(res_w) @ self.input_dp.ub
#                 + negative_only(res_w) @ self.input_dp.lb
#                 + res_b
#             )
#         else:
#             return self.prev_layer._get_ub(res_w, res_b)
#
# class DPReLU(torch.nn.Module, DPBackSubstitution):
#     """ DeepPoly transformer for ReLU layer """
#     def __init__(self) -> None:
#         super(DPReLU, self).__init__()
#
#     def forward(self):
#         pass
#
# class DPLinear(torch.nn.Module, DPBackSubstitution):
#     """ DeepPoly transformer for affine layer """
#     def __init__(self) -> None:
#         super().__init__()
#
#     def forward(self):
#         pass
#
#
# def negative_only(w):
#     return -torch.relu(-w)
#
#
# def positive_only(w):
#     return torch.relu(w)
    

def transform_image(pixel_values, input_dim):
    normalized_pixel_values = torch.tensor([float(p) / 255.0 for p in pixel_values])
    if len(input_dim) > 1:
        input_dim_in_hwc = (input_dim[1], input_dim[2], input_dim[0])
        image_in_hwc = normalized_pixel_values.view(input_dim_in_hwc)
        image_in_chw = image_in_hwc.permute(2, 0, 1)
        image = image_in_chw
    else:
        image = normalized_pixel_values

    assert (image >= 0).all()
    assert (image <= 1).all()
    return image

def get_spec(spec, dataset):
    input_dim = [1, 28, 28] if dataset == 'mnist' else [3, 32, 32]
    eps = float(spec[:-4].split('/')[-1].split('_')[-1])
    test_file = open(spec, "r")
    test_instances = csv.reader(test_file, delimiter=",")
    for i, (label, *pixel_values) in enumerate(test_instances):
        inputs = transform_image(pixel_values, input_dim)
        inputs = inputs.to(DEVICE).to(dtype=DTYPE)
        true_label = int(label)
    inputs = inputs.unsqueeze(0)
    return inputs, true_label, eps


def get_net(net, net_name):
    net = get_network(DEVICE, net)
    state_dict = torch.load('../nets/%s' % net_name, map_location=torch.device(DEVICE))
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict)
    net = net.to(dtype=DTYPE)
    net.eval()
    if 'resnet' in net_name:
        net = NormalizedResnet(DEVICE, net)
    return net


# convert networks to verifiable networks
def verifiable(net, pixels):
    verifiable_net = []
    if type(net) == NormalizedResnet:
        layers = [module for module in net.modules() if type(module) is ResNet]
        for modu in layers:
            for layer in modu:
                if type(layer) == nn.ReLU:
                    if len(verifiable_net) == 0:
                        verifiable_net.append(DPReLU(len(pixels)))
                    else:
                        verifiable_net.append(DPReLU(verifiable_net[-1].out_features))
                elif type(layer) == nn.Linear:
                    verifiable_net.append(DPLinear(layer))
                elif type(layer) == nn.Conv2d:
                    if len(verifiable_net) == 0:
                        verifiable_net.append(DPConv(layer, len(pixels)))
                    else:
                        verifiable_net.append(DPConv(layer, verifiable_net[-1].out_features))
                elif type(layer) == nn.Sequential:
                    for lay in layer:
                        if type(lay) == nn.ReLU:
                            if len(verifiable_net) == 0:
                                verifiable_net.append(DPReLU(len(pixels)))
                            else:
                                verifiable_net.append(DPReLU(verifiable_net[-1].out_features))
                        elif type(lay) == nn.Linear:
                            verifiable_net.append(DPLinear(lay))
                        elif type(lay) == nn.Conv2d:
                            if len(verifiable_net) == 0:
                                verifiable_net.append(DPConv(lay, len(pixels)))
                            else:
                                verifiable_net.append(DPConv(lay, verifiable_net[-1].out_features))
                        elif type(lay) == BasicBlock:
                            if len(verifiable_net) == 0:
                                verifiable_net.append(DPBasicBlock(lay, len(pixels)))
                            else:
                                verifiable_net.append(DPBasicBlock(lay, verifiable_net[-1].out_features))


    else:
        layers = [module for module in net.modules() if type(module) not in [FullyConnected, Conv, NormalizedResnet, nn.Sequential]]
        for layer in layers:
            # print(layer)
            if type(layer) == nn.ReLU:
                if len(verifiable_net) == 0:
                    verifiable_net.append(DPReLU(len(pixels)))
                else:
                    verifiable_net.append(DPReLU(verifiable_net[-1].out_features))
            elif type(layer) == nn.Linear:
                verifiable_net.append(DPLinear(layer))
            elif type(layer) == nn.Conv2d:
                if len(verifiable_net) == 0:
                    verifiable_net.append(DPConv(layer, len(pixels)))
                else:
                    verifiable_net.append(DPConv(layer, verifiable_net[-1].out_features))

    return nn.Sequential(*verifiable_net)


def normalize(value, inputs):
    if inputs.shape == torch.Size([1, 1, 28, 28]):
        norm = Normalization(DEVICE, 'mnist')
        norm_val = norm(value).view(-1)
        # norm_val = norm(value)
    else:
        norm = Normalization(DEVICE, 'cifar10')
        norm_val = norm(value).reshape(-1)
        # norm_val = norm(value)
    return norm_val

def analyze(net, inputs, eps, true_label):
    low_bound = normalize((inputs - eps).clamp(0, 1), inputs)
    up_bound = normalize((inputs + eps).clamp(0, 1), inputs)

    ####################################### new added
    # for alpha in range(10):
    #     verifiable_net = verifiable(net, inputs)
    #     optimizer = optim.Adam(verifiable_net.parameters(), lr=LR)
    #     for i in range(num_iter):
    #         optimizer.zero_grad()
    #         verifier_output = verifiable_net(DeepPoly(low_bound.shape[0], low_bound, up_bound))
    #         res = verifier_output.compute_verify_result(true_label)
    #         if (res > 0).all():
    #             break
    #         loss = torch.log(-res[res < 0]).max()
    #         # loss = (-res[res < 0]).max()
    #         loss.backward()
    #         optimizer.step()
    #         for p in verifiable_net.parameters():
    #             if p.requires_grad:
    #                 p.data.clamp_(0, 0.7854)
    #
    #     verifier_output = verifiable_net(DeepPoly(low_bound.shape[0], low_bound, up_bound))
    #     res = verifier_output.compute_verify_result(true_label)
    #     if (res > 0).all():
    #         continue
    #     else:
    #         return False
    # return True
    ####################################### new added

    verifiable_net = verifiable(net, inputs.reshape(-1))
    optimizer = optim.Adam(verifiable_net.parameters(), lr=LR)
    for i in range(num_iter):
        optimizer.zero_grad()
        verifier_output = verifiable_net(DeepPoly(low_bound.shape[0], low_bound, up_bound))
        res = verifier_output.compute_verify_result(true_label)
        if (res > 0).all():
            return True
        loss = torch.log(-res[res < 0]).max()
        loss.backward()
        optimizer.step()
        for p in verifiable_net.parameters():
            if p.requires_grad:
                p.data.clamp_(0, 0.7854)

    verifier_output = verifiable_net(DeepPoly(low_bound.shape[0], low_bound, up_bound))
    res = verifier_output.compute_verify_result(true_label)
    if (res > 0).all():
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net', type=str, required=True, help='Neural network architecture to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    net_name = get_net_name(args.net)
    dataset = 'mnist' if 'mnist' in net_name else 'cifar10'
    
    inputs, true_label, eps = get_spec(args.spec, dataset)
    net = get_net(args.net, net_name)
    # print([module for module in net.modules()])
    if type(net) == NormalizedResnet:
        print(net)
    for module in net.modules():
        if type(module) == ResNet:
            print("-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
            for layer in module:
                print(layer)
            for layer in module:
                if type(layer) == nn.Sequential:
                    for modu in layer:
                        if type(modu) == BasicBlock:
                            for mo in modu.modules():
                                if type(mo) == nn.Sequential:
                                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                                    print(mo)

    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label
    '''
    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')
    '''

if __name__ == '__main__':
    main()
