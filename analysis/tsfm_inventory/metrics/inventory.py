import numpy as np


def cost(order, demand, costs):
    leftover = np.maximum(order - demand, 0)
    shortfall = np.maximum(demand - order, 0)
    return costs.holding * leftover + costs.stockout * shortfall


def cost_per_unit(order, demand, costs):
    total_cost = cost(order, demand, costs).sum()
    total_demand = demand.sum()
    return float(total_cost / total_demand)


def fill_rate(order, demand):
    served = np.minimum(order, demand)
    return float(served.sum() / demand.sum())
