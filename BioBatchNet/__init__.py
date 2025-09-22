from .models.model import IMCVAE, GeneVAE
from .api import correct_batch_effects

# Dynamic version sourced from installed package metadata.
# Falls back to a placeholder when metadata is unavailable (e.g., running from source).
try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except Exception:  # pragma: no cover
    # Backport for older environments, though our requires-python is >=3.8
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("biobatchnet")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = ["IMCVAE", "GeneVAE", "correct_batch_effects"]
