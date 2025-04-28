from dataclasses import dataclass
import inspect
from typing import Optional


def get_generated_class(module_code, target_name):
    namespace = {}
    exec(module_code, namespace)
    for value in namespace.values():
        if inspect.isclass(value) and value.__name__ == target_name:
            return value
    raise ValueError(f'Class `{target_name}` not found in generated module')


def get_generated_function(module_code, target_name):
    namespace = {}
    exec(module_code, namespace)
    for value in namespace.values():
        if inspect.isfunction(value) and value.__name__ == target_name:
            return value
    raise ValueError(f'Function `{target_name}` not found in generated module')


def get_last_function(module_code):
    namespace = {}
    exec(module_code, namespace)
    last_func = None
    for value in namespace.values():
        if inspect.isfunction(value):
            last_func = value

    if last_func is None:
        raise ValueError(f'No functions found in the generated module')
    return last_func


@dataclass
class Adsorbent:
    name: str
    code: str
    method_of_synthesis: str

    def get_atoms(self):
        if not hasattr(self, '_get_atoms'):
            self._get_atoms = get_generated_function(self.code, 'create_adsorbent')
        return self._get_atoms()


@dataclass
class FAIRChemRelaxConfig:
    model_name: str = 'EquiformerV2-31M-S2EF-OC20-All+MD'
    model_local_cache: str = '/tmp/fairchem_checkpoints/'
    cpu: bool = True
    opt_fmax: float = 0.01
    opt_steps: int = 100
    checkpoint_path: Optional[str] = None
