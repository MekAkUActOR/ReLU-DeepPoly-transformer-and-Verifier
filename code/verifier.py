import argparse
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from networks import get_network, get_net_name, NormalizedResnet, FullyConnected, Normalization
from deeppoly import DeepPoly, DPReLU, DPLinear


DEVICE = 'cpu'
DTYPE = torch.float32
LR = 0.005
num_iter = 1000

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
    layers = [module for module in net.modules() if type(module) not in [FullyConnected, nn.Sequential]]
    verifiable_net = []

    for layer in layers:
        if type(layer) == nn.ReLU:
            if len(verifiable_net) == 0:
                verifiable_net.append(DPReLU(len(pixels)))
            else:
                verifiable_net.append(DPReLU(verifiable_net[-1].out_features))
        elif type(layer) == nn.Linear:
            verifiable_net.append(DPLinear(layer))

    return nn.Sequential(*verifiable_net)


def nomalize(value, inputs):
    if inputs.shape == torch.Size([1, 1, 28, 28]):
        norm = Normalization(DEVICE, 'mnist')
        norm_val = norm(value).view(-1)
    else:
        norm = Normalization(DEVICE, 'cifar10')
        norm_val = norm(value).reshape(-1)
    return norm_val

def analyze(net, inputs, eps, true_label):
    low_bound = nomalize((inputs - eps).clamp(0, 1), inputs)
    up_bound = nomalize((inputs + eps).clamp(0, 1), inputs)
    verifiable_net = verifiable(net, inputs)
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
    print("net: ", net)

    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
