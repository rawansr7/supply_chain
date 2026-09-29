import numpy as np


def cost(order, demand, costs):
    return costs.holding * np.maximum(order - demand, 0) + costs.stockout * np.maximum(demand - order, 0)


def cost_per_unit(order, demand, costs):
    return float(cost(order, demand, costs).sum() / demand.sum())


def fill_rate(order, demand):
    return float(np.minimum(order, demand).sum() / demand.sum())
