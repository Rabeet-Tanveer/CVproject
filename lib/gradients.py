import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Function, Variable


class VanillaGrad:
    def __init__(self, pretrained_model, cuda=False):
        self.model = pretrained_model
        self.model.eval()
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()

    def __call__(self, x, index=None):
        x.requires_grad = True
        output = self.model(x)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = torch.zeros_like(output)
        one_hot[0][index] = 1
        output.backward(gradient=one_hot)

        grad = x.grad.data[0].cpu().numpy()
        return grad


class SmoothGrad(VanillaGrad):
    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15, n_samples=25, magnitude=True):
        super().__init__(pretrained_model, cuda)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude

    def __call__(self, x, index=None):
        x_cpu = x.data.cpu().numpy()
        stdev = self.stdev_spread * (np.max(x_cpu) - np.min(x_cpu))
        total_gradients = np.zeros_like(x_cpu)

        for _ in range(self.n_samples):
            noise = np.random.normal(0, stdev, x_cpu.shape).astype(np.float32)
            x_noise = x_cpu + noise
            x_var = torch.tensor(x_noise, requires_grad=True)
            if self.cuda:
                x_var = x_var.cuda()

            output = self.model(x_var)
            if index is None:
                index = np.argmax(output.data.cpu().numpy())

            one_hot = torch.zeros_like(output)
            one_hot[0][index] = 1
            self.model.zero_grad()
            output.backward(gradient=one_hot)

            grad = x_var.grad.data.cpu().numpy()
            if self.magnitude:
                total_gradients += grad * grad
            else:
                total_gradients += grad

        avg_gradients = total_gradients[0] / self.n_samples
        return avg_gradients


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[grad_output < 0] = 0
        return grad_input


def replace_relu_with_guided(model):
    for name, module in model.features._modules.items():
        if isinstance(module, nn.ReLU):
            model.features._modules[name] = GuidedReLUWrapper()


class GuidedReLUWrapper(nn.Module):
    def forward(self, input):
        return GuidedBackpropReLU.apply(input)


class GuidedBackpropGrad(VanillaGrad):
    def __init__(self, pretrained_model, cuda=False):
        super().__init__(pretrained_model, cuda)
        replace_relu_with_guided(self.model)


class GuidedBackpropSmoothGrad(SmoothGrad):
    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15, n_samples=25, magnitude=True):
        super().__init__(pretrained_model, cuda, stdev_spread, n_samples, magnitude)
        replace_relu_with_guided(self.model)


class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.features = model.features
        self.target_layers = target_layers
        self.gradients = []

    def __call__(self, x):
        outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return outputs, x

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients


class GradCam:
    def __init__(self, pretrained_model, target_layer_names, cuda):
        self.model = pretrained_model
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.model.eval()
        self.extractor = FeatureExtractor(self.model, target_layer_names)

    def __call__(self, x, index=None):
        features, output = self.extractor(x)

        if index is None:
            index = np.argmax(output.data.cpu().numpy())

        one_hot = torch.zeros_like(output)
        one_hot[0][index] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1].cpu().data.numpy()[0]
        weights = np.mean(grads_val, axis=(2, 3))[0]

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
