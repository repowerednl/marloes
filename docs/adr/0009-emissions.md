# 9. Emissions

Date: 2025-01-31

## Status
Accepted

## Context
In our MARL energy management system, we aim to include the environmental impact of the used energy assets. In order to define a penalty for this impact, we have to define the **lifecycle GHG emissions** estimates for each of the relevant technologies:
- Photovoltaic (PV) solar
- Wind
- Battery storage
- Electrolysers
- Grid electricity (for the Dutch national grid)

We have gathered emissions data from multiple sources, this ADR formally records the references for the values we use.

## Decision
We will adopt the following lifecycle GHG emissions factors (in grams of CO₂-equivalent per kWh of electricity generated/consumed) for our current reward design:

1. **Photovoltaic (PV) Solar**
   - **Value:** ~45.5 gCO₂eq/kWh
   - **Rationale:**
     - The United Nations Economic Commission for Europe (UNECE) report [^1] cites a lifecycle range for European PV between 8–83 gCO₂eq/kWh.
     - We select **45.5 gCO₂eq/kWh** as a representative mid-range estimate.

2. **Wind**
   - **Value:** ~15.5 gCO₂eq/kWh
   - **Rationale:**
     - According to the UNECE study [^1], wind power ranges from 8–23 gCO₂eq/kWh (depending on onshore/offshore and specific site).
     - We choose **15.5 gCO₂eq/kWh** as a representative mid-range estimate.

3. **Battery**
   - **Value:** ~40–100 gCO₂eq/kWh
   - **Rationale:**
     - Production of lithium-ion batteries can vary widely, from 15–136 gCO₂eq/kWh for grid-scale systems [^2], and 61–106 gCO₂eq/kWh in [^3].
     - Since there is a very broad range, we assume a value between 40-100 gCO₂eq/kWh depending on the battery configuration.

4. **Electrolyser**
   - **Value:** Pending; determined by both the electrolyser technology and the source of electricity.
   - **Rationale:**
     - From [^4], the range for electrolyser GWP is:
       - Alkaline Water Electrolysis (AWE): 1–30 kgCO₂eq/kg H₂
       - Proton Exchange Membrane (PEMWE): 0.5–30 kgCO₂eq/kg H₂
       - Solid Oxide (SOEC): 0–5 kgCO₂eq/kg H₂
     - **Important:** For a more specific approximation, we will attempt to get specifications from Stoff2, the innovation that will be placed in this project by Reformers.

5. **Grid Electricity (Dutch Grid)**
   - **Value:** **284.73 gCO₂eq/kWh**
   - **Rationale:**
     - Calculated from CBS data on the **2023 Dutch energy mix** [^5]:
       - Wind: 29.166 B kWh @ 15.5 g
       - Solar: 19.993 B kWh @ 45.5 g
       - Biomass: 6.776 B kWh @ 49 g
       - Hydro: 0.068 B kWh @ 8.55 g
       - Natural Gas: 44.873 B kWh @ 458 g
       - Coal: 10.146 B kWh @ 923 g
     - Weighting these contributions yields a total average of **284.73 gCO₂eq/kWh** for the Dutch grid in 2023.

## Consequences
Defining these lifecycle GHG emission factors—and citing the references behind them—makes our reward calculations robust and grounded in current literature.

---

## References

[^1]: United Nations Economic Commission for Europe (UNECE). *Carbon Neutrality in the UNECE Region: Integrated Life-cycle Assessment of Electricity Sources*. 2022. [Link](https://unece.org/sites/default/files/2022-04/LCA_3_FINAL%20March%202022.pdf)
[^2]: Gutsch, Moritz & Leker, Jens. *Global warming potential of lithium-ion battery energy storage systems: A review*. *Journal of Energy Storage*. 2022. [DOI: 10.1016/j.est.2022.105030](https://doi.org/10.1016/j.est.2022.105030)
[^3]: Emilsson, Erik & Dahllöf, Lisbeth. *Lithium-Ion Vehicle Battery Production Status 2019*. Swedish Energy Agency. 2019. [DOI: 10.13140/RG.2.2.29735.70562](https://doi.org/10.13140/RG.2.2.29735.70562)
[^4]: Shaya, Negar & Glöser-Chahoud, Simon. *A Review of Life Cycle Assessment (LCA) Studies for Hydrogen Production Technologies through Water Electrolysis: Recent Advances*. *Energies*, 2024. [DOI: 10.3390/en17163968](https://doi.org/10.3390/en17163968)
[^5]: Statistics Netherlands (CBS). *Over half of electricity production now comes from renewable sources*. 2024. [Link](https://www.cbs.nl/en-gb/news/2024/39/over-half-of-electricity-production-now-comes-from-renewable-sources)

---
