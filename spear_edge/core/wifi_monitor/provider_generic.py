from __future__ import annotations

from .provider_kismet import KismetProvider


class GenericProvider(KismetProvider):
    """
    Generic command-backed provider.

    For v1 this shares the same command contract as the Kismet adapter so users
    can quickly swap providers via config without changing the manager.
    """

    name = "generic"
