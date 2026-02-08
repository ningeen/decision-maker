from mcda.core import MCDAMethod, MethodResult
from mcda.methods.ahp import AHPMethod
from mcda.methods.choix import ChoixMethod

METHODS = {
    "ahp": AHPMethod(),
    "choix": ChoixMethod(),
}


def get_method(method_id: str) -> MCDAMethod:
    return METHODS.get(method_id, METHODS["ahp"])


def list_methods() -> dict:
    return {method_id: method.name for method_id, method in METHODS.items()}
