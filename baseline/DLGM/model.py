import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence
from torch.distributions.utils import probs_to_logits
from torch.nn.utils import spectral_norm


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


class DLGM(nn.Module):
    def __init__(self, size, nx, nh, nz, step_size=0.005, n_steps=2):
        super(DLGM, self).__init__()
        self.encoder = Encoder(size, nx, nh, 2*nz)
        self.decoder = Decoder(size, nx, nh, nz)
        self.logit_tau = nn.Parameter(torch.zeros(1))
        self.nz = nz
        self.step_size = step_size
        self.n_steps = n_steps
        self.reset_parameters()

    def reset_parameters(self):
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]:
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):
        x = 2 * x - 1

        mu_z, logit_tau_z = self.encoder(x).chunk(2, 1)
        sigma_z = F.softplus(logit_tau_z).reciprocal().sqrt()
        q_z = Independent(Normal(mu_z, sigma_z), 1)
        z = q_z.rsample()
        p_z = Independent(Normal(torch.zeros_like(z), torch.ones_like(z)), 1)

        mu_x = self.decoder(z)
        sigma_x = F.softplus(self.logit_tau).reciprocal().sqrt().reshape(1, -1, 1, 1)

        energy = discretized_logistic_energy(x, mu_x, sigma_x).sum([1, 2, 3])
        kl = kl_divergence(q_z, p_z)

        loss = energy + kl

        if self.training:
            mu_x = self.decoder(z.detach())

            energy = discretized_logistic_energy(x, mu_x, sigma_x).sum([1, 2, 3])

            loss -= energy

            for t in range(self.n_steps):
                z = self.langevin_step(x, z)

            mu_x = self.decoder(z)

            energy = discretized_logistic_energy(x, mu_x, sigma_x).sum([1, 2, 3])
            loss += energy

        return loss

    def langevin_step(self, x, z):
        z.requires_grad_()

        mu_x = self.decoder(z)
        sigma_x = F.softplus(self.logit_tau).reciprocal().sqrt().reshape(1, -1, 1, 1)

        energy = discretized_logistic_energy(x, mu_x, sigma_x).sum([1, 2, 3])
        energy += z.pow(2).sum(1).div(2)
        grad = torch.autograd.grad(energy.sum(), z)[0]

        mu_z = z - self.step_size * grad
        sigma_z = math.sqrt(2*self.step_size)
        q = Independent(Normal(mu_z, sigma_z), 2)
        z = q.sample().detach()

        return z

    def sample(self, n=1):
        z = torch.randn(n, self.nz, device=next(self.parameters()).device)
        x = self.decoder(z)
        x = (x + 1) / 2

        return x.clamp(0, 1)

    def reconstruct(self, x):
        x = 2 * x - 1
        eps = torch.finfo(torch.float).eps

        mu_z, logit_tau_z = self.encoder(x).chunk(2, 1)
        sigma_z = F.softplus(logit_tau_z).reciprocal().sqrt()
        q_z = Independent(Normal(mu_z, sigma_z), 1)
        z = q_z.sample()

        x = self.decoder(z)
        x = (x + 1) / 2

        return x.clamp(0, 1)
