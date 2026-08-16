"""Returning the memory a finished matting used back to the operating system."""

from __future__ import annotations

import ctypes
import ctypes.util
import gc


def release_memory() -> None:
    """Collect garbage and hand freed arenas back to the OS.

    Matting allocates hundreds of megabytes of transient buffers. Freeing them inside the
    process is not enough on a 512MB host: glibc keeps the arenas, so the next run starts
    from an inflated RSS and gets OOM-killed. `malloc_trim` gives them back.
    """
    gc.collect()
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return
    try:
        libc = ctypes.CDLL(libc_name)
        libc.malloc_trim(0)
    except (OSError, AttributeError):  # not glibc (musl, macOS, Windows)
        pass
