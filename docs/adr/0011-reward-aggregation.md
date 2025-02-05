# 11. Reward Aggregation

Date: 2025-02-05

## Status

Accepted

## Context

Since we are dealing with a multi-objective reward, we should justify our way of combining/aggregating the rewards.

## Decision

As of right now, we do have a multi-objective reward (self-sufficiency and emissions), but since they are in no way competing and we conclude that the overall value of taking an action is just defined by the sum of the individual rewards. Therefore, we simply take the sum of both rewards.

## Consequences

None.
