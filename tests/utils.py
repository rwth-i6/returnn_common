"""
Test utils
"""


def assert_equal(a, b):
    """
    :param T a:
    :param T b:
    """
    assert type(a) == type(b), f"types differ: {type(a)} != {type(b)}"
    if a == b:
        return
    res = ["a != b", f"a: {a}", f"b: {b}"]
    if isinstance(a, set):
        assert isinstance(b, set)
        res += [f"a - b: {a - b}", f"b - a: {b - a}"]
    raise Exception("\n".join(res))
