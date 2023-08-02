from .build import build_loader as _build_loader


def build_loader(config, simmim=False, is_pretrain=False):
    return _build_loader(config)