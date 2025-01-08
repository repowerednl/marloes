# marloes
This Repowered project uses Multi-Agent Reinforcement Learning (MARL) to Optimize Energy Systems in an Energy Hub, aiming to reduce CO₂ emissions through efficient management of multiple energy commodities.

## Poetry

To download Poetry, you can use the following command:

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

For more detailed instructions, refer to the [Poetry documentation](https://python-poetry.org/docs/#installation).

Once Poetry is available, initialize the project with the following commands (all necessary packages are in `pyproject.toml`):

```sh
poetry install
```

For development, make sure to install the pre-commit hooks for cleaner commits and a clean repo.

```sh
pre-commit install
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

### Visualization
Alternatively, to view the metrics resulting from your experiment, the following code can be run:
```sh
python main.py --visualizer
```

This will prompt a simple interface which allows you to fill in the uid(s) of the experiments you want to visualize and select the metrics to plot. Leaving the uid field empty will automatically select the latest run experiment. Multiple experiments can be plotted against each other by filling in multiple uids seperated by commas.

## This project uses ADRs (Architecture Decision Records) to document choices that need justification.

### Installation
On linux, navigate to the folder you want the repository. It does not have to be in the same folder as marloes:
```
git clone https://github.com/npryce/adr-tools.git
cd adr-tools
```

The `adr-tools` repo does not provide a make install target, so install it manually:
```
sudo cp src/* /usr/local/bin
```

Make sure the scripts are executable:
```
sudo chmod +x /usr/local/bin/adr-*
```

Now add the repository's `src` to your `PATH` in your `~/.bashrc` or `/.zshrc`.
Reload shell configuration
```
source ~/.bashrc
```

On mac:
`brew install adr-tools`

Verify installation with
```
adr --version
```

### How to Use
1. ADRs are stored in `docs/adr`.
2. Create a new ADR: `adr new "Decision Title"`.
3. Link ADRs when one decision modifies or supersedes another:
   - Example: `adr link 0001 0002`
4. Review all decisions that still need justification: `grep -l "TBJ" docs/adr/*.md`
   - You can also filter on specific names by: `grep -l "TBJ" docs/adr/*.md | xargs grep -l "[NAME]"`

# Own use

You can use the separate packages for your own experiments.
