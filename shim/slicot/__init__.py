import warnings
warnings.warn(
    "The 'slicot' package has been renamed to 'ctrlsys'. "
    "Please run: pip install ctrlsys",
    DeprecationWarning, stacklevel=2
)
from ctrlsys import *  # noqa: F401,F403
from ctrlsys import __version__  # noqa: F401
