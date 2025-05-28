# 15. Battery-Parameters

Date: 2025-04-30

## Status

Accepted

## Context

The default values for batteries are adapted from an actual battery that will be placed at the site in Alkmaar. (AC-coupled, ESS Liquid Cooling Cabinet - ESS-TRENE)
The maximum efficiency of the battery was 0.98, but no information was provided about a general or mean efficiency. We used 0.90 instead which is a more accurate value.

## Decision

The following values are added to the default config in `BatteryAgent`.
- `max_power_in`: 125 kW
- `max_power_out`: 125 kW
- `efficiency`: 0.90
- `total_cycles`: 8000
The following were not found but are likely values:
- `max_state_of_charge`: 0.95
- `min_state_of_charge`: 0.05

## Consequences

Having some real-life default values. In reality you would like to provide as much information to have control.
