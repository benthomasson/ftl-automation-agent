
import os
import importlib


class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def resolve_modules_path_or_package(modules_path_or_package):
    if modules_path_or_package is None:
        return None
    modules_full_path = os.path.abspath(modules_path_or_package)
    if os.path.exists(modules_full_path) and os.path.isdir(modules_full_path):
        return modules_full_path
    else:
        try:
            modules_package = importlib.import_module(modules_path_or_package)
            return os.path.abspath(modules_package.__path__[0])
        except ModuleNotFoundError as e:
            print(e)
            return None
