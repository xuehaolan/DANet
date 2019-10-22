from __future__ import absolute_import

from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pr_conv2d, peak_stimulation


def pr_forward(self, x, class_threshold=0, peak_threshold=30):
    # classification network forwarding
    x = self.features(x)
    class_response_maps = self.classifier(x)

    # sub-pixel peak finding
    if self.sub_pixel_locating_factor > 1:
        class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor,
                                         mode='bilinear', align_corners=True)

    # aggregate responses from informative receptive fields estimated via class peak responses
    peak_list, aggregation = peak_stimulation(class_response_maps, win_size=self.win_size,
                                              peak_filter=self.peak_filter)

    # extract instance-aware visual cues, i.e., peak response maps
    assert class_response_maps.size(
        0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
    if peak_list is None:
        peak_list = peak_stimulation(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                     peak_filter=self.peak_filter)

    peak_response_maps = []
    valid_peak_list = []
    # peak backpropagation
    grad_output = class_response_maps.new_empty(class_response_maps.size())
    for idx in range(peak_list.size(0)):
        if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
            peak_val = class_response_maps[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
            if peak_val > peak_threshold:
                grad_output.zero_()
                # starting from the peak
                grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
                if input.grad is not None:
                    input.grad.zero_()
                class_response_maps.backward(grad_output, retain_graph=True)
                prm = input.grad.detach().sum(1).clone().clamp(min=0)
                peak_response_maps.append(prm / prm.sum())
                valid_peak_list.append(peak_list[idx, :])

    # return results
    class_response_maps = class_response_maps.detach()
    aggregation = aggregation.detach()

    if len(peak_response_maps) > 0:
        valid_peak_list = torch.stack(valid_peak_list)
        peak_response_maps = torch.cat(peak_response_maps, 0)
        return aggregation, class_response_maps, valid_peak_list, peak_response_maps
    else:
        return None


def pr_wrap(net):
    net.eval()

    # modify the backward method of conv2d
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            module._original_forward = module.forward
            module.forward = MethodType(pr_conv2d, module)

    # modify the forward method of net
    net._original_forward = net.forward
    net.forward = MethodType(pr_forward, net)

    return net


def pr_unwrap(net):
    # recover each conv2d layer
    for module in net.modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
            module.forward = module._original_forward

    # recover the net forward method
    if hasattr(net, '_original_forward'):
        net.forward = net._original_forward
