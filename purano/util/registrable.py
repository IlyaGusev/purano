# Based on https://github.com/allenai/allennlp/blob/master/allennlp/common/registrable.py
from collections import defaultdict


class Registrable:
    _registry = defaultdict(dict)

    @classmethod
    def register(cls, name):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass):
            if name in registry:
                message = (
                    f"Cannot register {name} as {cls.__name__}; "
                    f"name already in use for {registry[name][0].__name__}"
                )
                raise RuntimeError(message)
            registry[name] = (subclass)
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls, name: str):
        if name in Registrable._registry[cls]:
            return Registrable._registry[cls].get(name)
        raise RuntimeError(f"{name} is not a registered name for {cls.__name__}. ")
