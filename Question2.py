from types import SimpleNamespace

class ErrorMarketClass:
    def __init__(self):
        par = self.par = SimpleNamespace()
        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3
        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.p2 = 1
        self.N = 75

    def utility_A(self, x1A, x2A):
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    def utility_B(self, x1B, x2B):
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    def demand_A(self, p1):
        x1A_star = self.par.alpha * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / p1
        x2A_star = (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.p2 * self.par.w2A) / self.par.p2
        return x1A_star, x2A_star

    def demand_B(self, p1):
        x1B_star = self.par.beta * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / p1
        x2B_star = (1 - self.par.beta) * (p1 * (1 - self.par.w1A) + self.par.p2 * (1 - self.par.w2A)) / self.par.p2
        return x1B_star, x2B_star

    def check_market_clearing(self):
        par = self.par
        self.rho1 = [0.5 + 2 * i / self.N for i in range(self.N + 1)]

        errors = []

        for p1 in self.rho1:
            x1A_star, x2A_star = self.demand_A(p1)
            x1B_star, x2B_star = self.demand_B(p1)
            eps1 = x1A_star - par.w1A + x1B_star - (1 - par.w1A)
            eps2 = x2A_star - par.w2A + x2B_star - (1 - par.w2A)
            errors.append((eps1, eps2))

        return errors

errormarket = ErrorMarketClass()
result = errormarket.check_market_clearing()

print("Errors in the market clearing condition:")
for eps1, eps2 in result:
    print(f"Error 1: {eps1}, Error 2: {eps2}")
