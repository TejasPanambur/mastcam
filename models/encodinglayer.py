import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair

__all__ = ['View', 'Normalize', 'Encoding']

class Encoding(Module):
    
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        
        X = X.unsqueeze(2).expand(X.size(0), X.size(1),self.codewords.size(0), self.codewords.size(1)) 
        C = self.codewords.unsqueeze(0).unsqueeze(0)
        S = self.scale.view(1, 1, self.codewords.size(0)) 
        # assignment weights BxNxK
        A = F.softmax(S*((X-C).pow(2).sum(3)), dim=2)
        A = A.unsqueeze(3)
        # aggregate
        E = (A * (X-C)).sum(1)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'
    
    
    
class View(nn.Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """
    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)
    

class Normalize(nn.Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.
    Does:
    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}
    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.
    With default arguments normalizes over the second dimension with Euclidean
    norm.
    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)

    
