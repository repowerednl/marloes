# marloes
This Repowered project uses Multi-Agent Reinforcement Learning (MARL) to Optimize Energy Systems in an Energy Hub, aiming to reduce CO₂ emissions through efficient management of multiple energy commodities.

# Poetry

To download Poetry, you can use the following command:

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

For more detailed instructions, refer to the [Poetry documentation](https://python-poetry.org/docs/#installation).

Once Poetry is available, initialize the project with the following commands (all necessary packages are in `pyproject.toml`):

```sh
poetry install
```

To activate the virtual environment created by Poetry, use:

```sh
poetry shell
```

You can now run the project using:

```sh
python main.py
```

This will prompt a experiment startup screen for you to select a configuration to run a grid search, or an experiment with your personal preferences.

# Own use

You can use the separate packages for your own experiments.
