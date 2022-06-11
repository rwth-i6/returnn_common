"""
Test utils
"""


def assert_equal(a, b):
  """
  :param T a:
  :param T b:
  """
  assert a == b, f"{a!r} != {b!r}"
