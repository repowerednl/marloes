# 10. Self-Sufficiency reward

Date: 2025-02-05

## Status

Accepted

## Context

The primary goal of the REFORMERS project is to be self-sufficient over a year, which means to produce as much as is consumed. This is quite hard to define in a reward model, since it is a sparse reward (only available at the end of the year). We need to define intermediate rewards to achieve the self-sufficiency goal. We already implemented the cumulative_grid_state, which is a measure of how much energy is taken from (or fed into) the grid. Producing as much as possible every day to meet your own demand and, if possible, feed into the grid would lead to the optimal result. If you are self-sufficient every day, you are self-sufficient every year.

## Decision

As immediate for reward we use the cumulative grid state as an indication of what should be done to achieve the goal. If we take from the grid, the model is penalized whereas producing more will not be penalized (reward of 0). This way we would continue producing in the summer, potentially compensating for winter months.

## Consequences

No changes to the code are needed. To achieve possitive Self-Sufficiency, which would not be bad either, we should remove the max(0,cum_state_grid) operator in the reward calculation. This is however exactly what Net Balance (NB sub reward) is.
