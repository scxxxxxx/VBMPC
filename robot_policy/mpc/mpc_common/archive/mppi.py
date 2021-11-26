import torch
from torch.distributions import Normal

class MPPI(object):
    def __init__(self, model, samples=40, horizon=10, lam=0.1, eps=0.2, device=None):
        self.model = model
        self.a_dim = model.a_dim
        self.horizon = horizon
        self.lam = lam
        self.samples = samples

        self.device = device

        self.a = torch.zeros((horizon, self.a_dim), device=self.device)
        # mu = torch.zeros((self.samples, self.num_actions), device=self.device)
        mu = torch.randn((self.samples, self.a_dim), device=self.device)
        sigma = torch.ones((self.samples, self.a_dim), device=self.device) * eps
        self.eps = Normal(mu, sigma)

    def reset(self):
        self.a.zero_()

    def update(self, state):

        with torch.no_grad():
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()

            s0 = torch.tensor(state, device=self.device).float().unsqueeze(0)
            s = s0.repeat(self.samples, 1)

            sk, da, log_prob = [], [], []
            eta = None
            gamma = 0.5
            for t in range(self.horizon):
                eps = self.eps.sample()
                eta = eps
                # if eta is None:
                #     eta = eps
                # else:
                #     eta = gamma*eta + ((1-gamma**2)**0.5) * eps
                v = self.a[t].expand_as(eta) + eta
                # ic(v, v.shape)
                # exit()
                s, rew = self.model.step(s, v)
                log_prob.append(self.eps.log_prob(eta).sum(1))
                da.append(eta)
                sk.append(rew.squeeze())

            sk = torch.stack(sk)
            sk = torch.cumsum(sk.flip(0), 0).flip(0)
            log_prob = torch.stack(log_prob)

            sk = sk + self.lam*log_prob
            sk = sk - torch.max(sk, dim=1, keepdim=True)[0]
            w = torch.exp(sk.div(self.lam)) + 1e-5
            w.div_(torch.sum(w, dim=1, keepdim=True))
            for t in range(self.horizon):
                self.a[t] = self.a[t] + torch.mv(da[t].T, w[t])

            return self.a[0].cpu().clone().numpy()
