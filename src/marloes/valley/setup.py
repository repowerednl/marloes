from marloes.valley import env


def setup_env(config: dict):
    # pass necessary config to the model
    model = env.EnergyValley(config)
    return model
