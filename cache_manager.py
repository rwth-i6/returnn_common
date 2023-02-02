"""
Helpers for the RWTH-i6 cache manager

https://github.com/rwth-i6/cache-manager
"""

import os
from subprocess import check_output, CalledProcessError


def cf(filename: str) -> str:
    """Cache manager"""
    if filename in _cf_cache:
        return _cf_cache[filename]
    debug_mode = _get_debug_mode()
    hostname = _get_hostname()
    if debug_mode or not hostname.startswith("cluster-cn-"):
        print("use local file: %s" % filename)
        return filename  # for debugging
    try:
        cached_fn = check_output(["cf", filename]).strip().decode("utf8")
    except CalledProcessError:
        print("Cache manager: Error occurred, using local file")
        return filename
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn


_cf_cache = {}


def _get_debug_mode() -> bool:
    from returnn.config import get_global_config

    config = get_global_config()
    return config.typed_dict.get("debug_mode", False)


def _get_hostname() -> str:
    return check_output(["hostname"]).strip().decode("utf8")
