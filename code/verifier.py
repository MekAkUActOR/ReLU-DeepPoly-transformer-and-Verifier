import argparse
import csv
import torch
import torch.nn.functional as F
from networks import get_network, get_net_name, NormalizedResnet


DEVICE = 'cpu'
DTYPE = torch.float32


class DeepPoly:
    def __init__(self, lb, ub, lexpr, uexpr) -> None:
        self.lb = lb
        self.ub = ub
        self.lexpr = lexpr
        self.uexpr = uexpr
        assert not torch.isnan(self.lb).any()
        assert not torch.isnan(self.ub).any()
        assert lexpr is None or (
            (not torch.isnan(self.lexpr[0]).any())
            and (not torch.isnan(self.lexpr[1]).any())
        )
        assert uexpr is None or (
            (not torch.isnan(self.uexpr[0]).any())
            and (not torch.isnan(self.uexpr[1]).any())
        )
        self.dim = lb.size()[0]
    
    @staticmethod
    def deeppoly_from_perturbation(x, eps):
        assert eps >= 0, "epsilon must not be negative value"
        # We only have to verify images with pixel intensities between 0 and 1.
        # e.g. if perturbation is 0.2 and pixel value is 0.9
        # we verify range of [0.7， 1.0], instead of [0.7, 1.1]
        lb = x - eps
        ub = x + eps
        lb[lb < 0] = 0
        ub[ub > 1] = 1
        return DeepPoly(lb, ub, None, None)
    


class DPBackSubstitution:
    def __init__(self) -> None:
        pass

    def _get_lb(self, expr_w, expr_b):
        if len(self.output_dp.lexpr[0].size()) == 2:
            res_w = (
                positive_only(expr_w) @ self.output_dp.lexpr[0]
                + negative_only(expr_w) @ self.output_dp.uexpr[0]
            )
        else:
            res_w = (
                positive_only(expr_w) * self.output_dp.lexpr[0]
                + negative_only(expr_w) * self.output_dp.uexpr[0]
            )
        res_b = (
            positive_only(expr_w) @ self.output_dp.lexpr[1]
            + negative_only(expr_w) @ self.output_dp.uexpr[1]
            + expr_b
        )

        if self.prev_layer == None:
            return (
                positive_only(res_w) @ self.input_dp.lb
                + negative_only(res_w) @ self.input_dp.ub
                + res_b
            )
        else:
            return self.prev_layer._get_lb(res_w, res_b)

    def _get_ub(self, expr_w, expr_b):
        if len(self.output_dp.lexpr[0].size()) == 2:
            res_w = (
                positive_only(expr_w) @ self.output_dp.uexpr[0]
                + negative_only(expr_w) @ self.output_dp.lexpr[0]
            )
        else:
            res_w = (
                positive_only(expr_w) * self.output_dp.uexpr[0]
                + negative_only(expr_w) * self.output_dp.lexpr[0]
            )
        res_b = (
            positive_only(expr_w) @ self.output_dp.uexpr[1]
            + negative_only(expr_w) @ self.output_dp.lexpr[1]
            + expr_b
        )

        if self.prev_layer == None:
            return (
                positive_only(res_w) @ self.input_dp.ub
                + negative_only(res_w) @ self.input_dp.lb
                + res_b
            )
        else:
            return self.prev_layer._get_ub(res_w, res_b)


def negative_only(w):
    return -torch.relu(-w)


def positive_only(w):
    return torch.relu(w)    
    

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


def analyze(net, inputs: torch.Tensor, eps: torch.float32, true_label: int):
    """ Analyze the neural net's robustness with DeepPoly. 
    Arguments
    ---------
    net : nn.Module
        will be a Fully Connected, CNN or Residual Neural Net
    inputs : torch.Tensor
        input image to the neural net
    eps : torch.float32
        epsilon value
    true label : int
        ground truth label
        
    Return
    ------
    Boolean value, True if robustness can be verified
    """
    print(net.layers)
    print(eps)
    print(inputs.shape)
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net', type=str, required=True, help='Neural network architecture to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    net_name = get_net_name(args.net)
    dataset = 'mnist' if 'mnist' in net_name else 'cifar10'
    
    inputs, true_label, eps = get_spec(args.spec, dataset)
    net = get_net(args.net, net_name)

    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
