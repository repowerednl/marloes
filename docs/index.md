<!-- Marloes master file, manually created on March
   18, 2025. Automatically creates full html when running
   "make html" in the docs folder. -->

Welcome to {{Package}}'s documentation!
=======================================
```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   marloes
```

```{include} ../README.md
:start-line: 1
:end-line: 17
```

```{eval-rst}
.. note::
   This way the library can be extended with new types of assets, and the interface for the assets is kept consistent. The split between the asset and the state allows for the state to be saved and loaded, and the simulation to be resumed from that point in time, which is useful for persistence in real-time simulations.
```

```{include} ../README.md
:start-line: 19
```
