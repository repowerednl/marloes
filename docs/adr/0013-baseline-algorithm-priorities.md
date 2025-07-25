# 12. Baseline Algorithm (priorities)

Date: 2025-02-14

## Status

Accepted

## Context

We need a baseline algorithm to compare the trained algorithms with. The easiest problem (distributing energy flows without markets) can be done through rule-based priorities. This is exactly what Simon does, the only issue is that batteries need setpoints to 'act' in Simon. For the Priorities algorithm we need a rule-based system to manually set a setpoint for the battery in Simon.

## Decision

The basic rule is; if there is more energy available (supply assets like solar/wind) than demand, we charge the batteries. If there is not enough energy, the batteries should discharge. Some restrictions can be applied, like a State of Charge threshold before discharging.

## Consequences

Some basic functionality for the batteries in Simon to work with priorities. This is not done for ElectrolyserAgent yet, but should work the same way.
