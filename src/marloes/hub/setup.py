from marloes.hub import env


def setup_env(config: dict):
    # pass necessary config to the model
    model = env.EnergyHub(config)
    return model
