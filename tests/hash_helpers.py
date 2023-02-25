"""
Persistent hash helpers (unlike Python builtin `hash`)

Copied over from Sisyphus (https://github.com/rwth-i6/sisyphus/blob/master/sisyphus/hash.py),
and then adapted and simplified for our use case here.
"""

import hashlib
from returnn_common import nn

# noinspection PyProtectedMember
from returnn.tf.util.data import _MarkedDim


def short_hash(obj, length=12, chars="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):
    """
    :param object obj:
    :param int length:
    :param str|T chars:
    :rtype: str|T
    """
    h = hashlib.sha256(hash_helper(obj)).digest()
    h = int.from_bytes(h, byteorder="big", signed=False)
    ls = []
    for i in range(length):
        ls.append(chars[int(h % len(chars))])
        h = h // len(chars)
    return "".join(ls)


def hash_helper(obj):
    """
    Takes most object and tries to convert the current state into bytes.

    :param object obj:
    :rtype: bytes
    """

    # Store type to ensure it's unique
    byte_list = [type(obj).__qualname__.encode("utf8")]

    # Using type and not isinstance to avoid derived types
    if isinstance(obj, bytes):
        byte_list.append(obj)
    elif obj is None:
        pass
    elif type(obj) in (int, float, bool, str, complex):
        byte_list.append(repr(obj).encode("utf8"))
    elif isinstance(obj, (list, tuple)):
        byte_list += map(hash_helper, obj)
    elif isinstance(obj, (set, frozenset)):
        byte_list += sorted(map(hash_helper, obj))
    elif isinstance(obj, dict):
        # sort items to ensure they are always in the same order
        byte_list += sorted(map(hash_helper, obj.items()))
    elif isinstance(obj, nn.Dim):
        byte_list.append(_hash_helper_dim(obj))
    elif isinstance(obj, _MarkedDim):
        byte_list.append(_hash_helper_marked_dim(obj))
    else:
        # Fail explicitly such that we handle everything explicitly.
        # Maybe we can loosen this restriction later.
        # The original Sisyphus code was more generic here.
        raise TypeError(f"unexpected object type {type(obj)}")

    byte_str = b"(" + b", ".join(byte_list) + b")"
    if len(byte_str) > 4096:
        # hash long outputs to avoid arbitrary long return values. 4096 is just
        # picked because it looked good and not optimized,
        # it's most likely not that important.
        return hashlib.sha256(byte_str).digest()
    else:
        return byte_str


def _hash_helper_dim(dim: nn.Dim) -> bytes:
    # also see Dim.__hash__
    if dim.special:
        if dim == nn.single_step_dim:
            return b"nn.single_step_dim"
        raise ValueError(f"Hash for special dim tag {dim} is not defined.")
    if dim.is_batch_dim():
        return b"nn.batch_dim"
    base = dim.get_same_base()
    if base is not dim:
        return _hash_helper_dim(base)
    if dim.derived_from_op:
        # noinspection PyProtectedMember
        return hash_helper((b"nn.Dim",) + dim.derived_from_op._value())
    return hash_helper((b"nn.Dim", base.kind.name, base.dimension, base.description))


def _hash_helper_marked_dim(dim: _MarkedDim) -> bytes:
    return hash_helper((dim.__class__.__name__.encode("utf8"), dim.tag))
