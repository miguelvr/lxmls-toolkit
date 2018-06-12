import torch
import numpy as np
import torch.nn.functional as F
from lxmls.deep_learning.utils import Model, glorot_weight_init
from lxmls.deep_learning.pytorch_models.utils import \
    cast_torch_int, cast_torch_float


class PytorchLogLinear(Model):

    def __init__(self, **config):

        super(PytorchLogLinear, self).__init__(**config)

        # Initialize parameters
        weight_shape = (config['input_size'], config['num_classes'])
        # after Xavier Glorot et al
        self.weight = glorot_weight_init(weight_shape, 'softmax')
        self.bias = np.zeros((1, config['num_classes']))
        self.learning_rate = config['learning_rate']

        # IMPORTANT: Cast to pytorch format
        self.weight = cast_torch_float(self.weight, requires_grad=True)
        self.bias = cast_torch_float(self.bias, requires_grad=True)

    def _log_forward(self, input=None):
        """Forward pass of the computation graph in
        logarithm domain (pytorch)"""

        # IMPORTANT: Cast to pytorch format
        input = cast_torch_float(input, requires_grad=False)

        # Linear transformation
        z =  torch.matmul(input, torch.t(self.weight)) + self.bias

        # Softmax implemented in log domain
        log_tilde_z = F.log_softmax(z)

        # NOTE that this is a pytorch class!
        return log_tilde_z

    def predict(self, input=None):
        """Most probably class index"""
        log_forward = self._log_forward(input).detach().numpy()
        return np.argmax(np.exp(log_forward), axis=1)

    def update(self, input=None, output=None):
        """Stochastic Gradient Descent update"""

        # IMPORTANT: Class indices need to be casted to LONG
        true_class = cast_torch_int(output, requires_grad=False)

        # Compute negative log-likelihood loss
        loss = torch.nn.NLLLoss()(self._log_forward(input), true_class)
        # Use autograd to compute the backward pass.
        loss.backward()

        with torch.no_grad():
            # SGD update
            self.weight.data -= self.learning_rate * self.weight.grad.data
            self.bias.data -= self.learning_rate * self.bias.grad.data

            # Zero gradients
            self.weight.grad.data.zero_()
            self.bias.grad.data.zero_()

        return loss.item()
