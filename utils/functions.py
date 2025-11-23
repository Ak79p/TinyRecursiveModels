import importlib
import inspect


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)


def apply_alpha_blend(prev_tensor, new_tensor, alpha):
    """
    Interpolate between previous and new tensors using alpha blending.
    
    Args:
        prev_tensor: Tensor or None (same shape as new_tensor), previous state
        new_tensor: Tensor (same shape), new prediction
        alpha: float in [0.0, 1.0], blending factor
    
    Returns:
        blended: Tensor, same dtype & device as new_tensor
    """
    if prev_tensor is None:
        return new_tensor
    if alpha == 1.0:
        return new_tensor
    return (1.0 - alpha) * prev_tensor + alpha * new_tensor
