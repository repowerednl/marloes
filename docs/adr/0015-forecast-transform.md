# 15. Forecast transform

Date: 2025-04-01

## Status

Accepted

## Context

The length of the forecast was based on `horizon`, which is quite large because we are working on minute-based data. If looking at a forecast for 1 day we get 1440 entries. This is unnecessarily expensive.


## Decision

Instead of `len(forecast)` elements we reduce it to 4:
- mean
- std
- average slope
- turning points (changing slope)

## Consequences

Forecast complexity reduced, reducing the elements in the agent states and thus the observation space.
