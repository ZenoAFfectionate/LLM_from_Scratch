try:
    import importlib.metadata
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"  # fallback version
