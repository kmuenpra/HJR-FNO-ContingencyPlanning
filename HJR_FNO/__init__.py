"""
HJR_FNO package
Provides Hamilton-Jacobi Reachability with Fourier Neural Operators
"""

# Expose classes but delay actual import to avoid circular dependency
__all__ = ["HJR_FNO", "Grid", "SpectralConv1d", "FNO1d"]

def __getattr__(name):
    """Lazy loading to avoid circular imports"""
    if name in __all__:
        from .HJR_FNO import HJR_FNO, Grid, SpectralConv1d, FNO1d
        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")