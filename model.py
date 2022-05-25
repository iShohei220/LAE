import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence, Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions.utils import probs_to_logits
from torch.nn.utils import spectral_norm, weight_norm


def discretized_logistic_energy(x, mu_x, sigma_x):
    eps = torch.finfo(torch.float).eps

    x_ceil = x + 1. / 255.
    x_floor = x - 1. / 255.
    c_ceil = (x_ceil - mu_x) / sigma_x
    c_floor = (x_floor - mu_x) / sigma_x

    energy = - torch.log(torch.clamp(c_ceil.sigmoid() - c_floor.sigmoid(), min=eps))
    energy_plus = F.softplus(c_floor)
    energy_minus = F.softplus(c_ceil) - c_ceil
    energy = torch.where(x_ceil > 1., energy_plus, energy)
    energy = torch.where(x_floor < -1., energy_minus, energy)

    return energy


class Encoder(nn.Module):
    def __init__(self, size, nx, nh, nz):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nx*size**2, nh),
            nn.LayerNorm(nh),
            nn.ReLU(True),
            nn.Linear(nh, nh),
            nn.LayerNorm(nh),
            nn.ReLU(True),
            nn.Linear(nh, nh),
            nn.LayerNorm(nh),
            nn.ReLU(True),
            nn.Linear(nh, nz)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, size, nx, nh, nz):
        super(Decoder, self).__init__()
        net = [
            nn.Linear(nz, nh),
            nn.LayerNorm(nh),
            act,
            nn.Linear(nh, nh),
            nn.LayerNorm(nh),
            act,
            nn.Linear(nh, nh),
            nn.LayerNorm(nh),
            act,
            nn.Linear(nh, nx*size**2),
            nn.Unflatten(1, (nx, size, size))
        ]
        self.net = nn.Sequential(*net)

    def forward(self, z):
        return self.net(z)


class LAE(nn.Module):
    def __init__(self, size, nx, nh, nz, step_size, n_steps=2, mh=True):
        super(LAE, self).__init__()
        self.encoder = Encoder(size, nx, nh, nz)
        self.decoder = Decoder(size, nx, nh, n)
        self.logit_tau = nn.Parameter(torch.zeros(1))
        self.nz = nz
        self.step_size = step_size
        self.n_steps = n_steps
        self.mh = mh

        self.reset_parameters()
        self.encoder.net[-1].weight.requires_grad_(False)

    def reset_parameters(self):
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]:
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):
        if self.training:
            x = 2 * x - 1

            energy = 0
            h = self.encoder.net[:-1](x)
            weight = self.encoder.net[-1].weight.data
            weight, _ = self.langevin_step(x, h, weight,
                                           self.mh, True)
            for i in range(self.n_steps):
                weight, energy_ = self.langevin_step(x, h, weight, True)
                energy += energy_

            energy /= self.n_steps
            self.encoder.net[-1].weight.data = weight.detach()

            return energy

        else:
            x = 2 * x - 1
            mu_z = self.encoder(x)
            q_z = Independent(Normal(mu_z, 0.05), 1)
            z = q_z.sample()
            p_z = Independent(Normal(torch.zeros_like(z), torch.ones_like(z)), 1)
            mu_x = self.decoder(z)
            sigma_x = F.softplus(self.logit_tau).reciprocal().sqrt().reshape(1, -1, 1, 1)

            energy = discretized_logistic_energy(x, mu_x, sigma_x).sum([1, 2, 3])
            kl = kl_divergence(q_z, p_z)
            loss = energy + kl

            return loss

    def langevin_step(self, x, h,
                      weight=None,
                      retain_graph=False):
        if weight is None:
            weight = self.encoder.net[-1].weight

        weight.requires_grad_()

        z = F.linear(h, weight)
        mu_x = self.decoder(z)
        sigma_x = F.softplus(self.logit_tau).reciprocal().sqrt().reshape(1, -1, 1, 1)

        energy = discretized_logistic_energy(x, mu_x, sigma_x).sum([1, 2, 3])
        energy += z.pow(2).sum(1).div(2)
        grad = torch.autograd.grad(energy.mean(), weight,
                                   retain_graph=retain_graph)[0]

        mu_w = weight.data - self.step_size * grad
        sigma_w = math.sqrt(2*self.step_size / x.size(0))
        q = Independent(Normal(mu_w, sigma_w), 2)
        weight_ = q.sample().detach()

        if self.mh:
            weight_.requires_grad_()
            z_ = F.linear(h.detach(), weight_)
            mu_x = self.decoder(z_)

            energy_ = discretized_logistic_energy(x, mu_x, sigma_x).sum([1, 2, 3])

            energy_ += z_.pow(2).sum(1).div(2)
            grad_ = torch.autograd.grad(energy_.mean(), weight_)[0]

            mu_w_ = weight_.data - self.step_size * grad_
            sigma_w = math.sqrt(2*self.step_size / x.size(0))
            q_ = Independent(Normal(mu_w_, sigma_w), 2)

            log_ratio = energy.sum() - energy_.sum() \
                        + q_.log_prob(weight) - q.log_prob(weight_)
            ratio = log_ratio.exp()

            acceptance_rate = torch.min(torch.ones_like(ratio), ratio)

            u = torch.rand_like(acceptance_rate)
            if u < acceptance_rate:
                weight = weight_.detach()
            else:
                weight = weight.detach()
        else:
            weight = weight_.detach()

        return weight, energy

    def sample(self, n=1):
        z = torch.randn(n, self.nz, device=next(self.parameters()).device)
        x = self.decoder(z)
        x = (x + 1) / 2

        return x.clamp(0, 1)

    def reconstruct(self, x):
        x = 2 * x - 1
        z = self.encoder(x)
        x = self.decoder(z)
        x = (x + 1) / 2

        return x.clamp(0, 1)
