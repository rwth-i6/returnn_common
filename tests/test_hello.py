
from . import _setup_test_env  # noqa
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_greater, assert_true, assert_false


def test_hello():
  from returnn_common.tests.hello import hello
  assert_equal(hello(), "hello world")

