import yaml

def load_config(path_config):
    with open(path_config, "r", encoding="utf-8") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    # print(args)
    return args

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__