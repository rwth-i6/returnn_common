
from . import _setup_test_env  # noqa
from nose.tools import assert_equal


def test_hello():
  from returnn_common.tests.hello import hello
  assert_equal(hello(), "hello world")

