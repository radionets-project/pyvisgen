from pyvisgen.io import Config
from pyvisgen.utils.carbon_tracking import carbontracker


def test_carbontracker():
    config = Config()  # use default values

    # codecarbon = False
    with carbontracker(config=config):
        pass

    config.codecarbon = True
    config = config.model_validate(config.model_dump())

    # codecarbon = True
    with carbontracker(config=config):
        pass
