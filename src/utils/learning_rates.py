import math


class CosineAnnealingWarmRestarts:

    def __init__(self, T_0 = 10, T_mult = 2, eta_min = 0.01, eta_max = 0.05, last_epoch = -1, max_epoch=math.inf):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_cur = last_epoch
        self.last_epoch = last_epoch
        self.max_epoch = max_epoch

    def get_lr(self):
        return self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2

    def step(self, epoch = None):

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                if self.max_epoch >= self.T_i * self.T_mult:
                    self.T_cur = self.T_cur - self.T_i
                    self.max_epoch -= self.T_i
                    self.T_i = self.T_i * self.T_mult
                else:
                    self.T_cur = self.T_cur - 1  # We prevent the learning rate to increase
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
