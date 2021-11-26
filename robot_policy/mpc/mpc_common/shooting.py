import torch


class Shooting(object):
    def __init__(self, model, T=10, lr=0.02, **kwargs):
        super(Shooting, self).__init__()
        self.T = T
        self.model = model

        self.state_dim = model.s_dim
        self.a_dim = model.a_dim

        self.device = model.device
        self.lr = lr
        self.u = torch.zeros((T, self.a_dim), device=self.device, requires_grad=True)

        self.optimizer = torch.optim.SGD([self.u], lr=lr)

    def reset(self):
        with torch.no_grad():
            self.u.zero_()

    def update(self, state, epochs=2):
        for epoch in range(epochs):
            s = torch.tensor(state, device=self.device).float().unsqueeze(0)
            cost = 0.
            for u in self.u:
                s, r = self.model.step(s, torch.tanh(u.unsqueeze(0)))
                cost = cost - r

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

        with torch.no_grad():
            u = torch.tanh(self.u[0]).detach().cpu().numpy()
            self.u[:-1] = self.u[1:].clone()
            self.u[-1].zero_()
            return u
