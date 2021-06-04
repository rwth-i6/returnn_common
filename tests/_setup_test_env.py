"""
Setups the environment for tests.
In the test code, you would have this::

    from . import _setup_test_env  # noqa

This is a relative import because `tests` is a package.

Also see :mod:`_setup_path`.
See :func:`setup` below for details.
"""

import sys
import os

my_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
root_dir = os.path.dirname(my_dir)
parent_root_dir = os.path.dirname(root_dir)
_my_old_mod_name = __name__
_expected_package_name = "returnn_common.tests"
_expected_mod_name = _expected_package_name + "._setup_test_env"


def setup():
  """
  Calls necessary setups.
  """
  import logging

  # Enable all logging, up to debug level.
  logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s')

  # Disable extensive TF debug verbosity. Must come before the first TF import.
  logging.getLogger('tensorflow').disabled = True
  # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
  # logging.getLogger("tensorflow").setLevel(logging.INFO)

  # Get us some further useful debug messages (in some cases, e.g. CUDA).
  # For example: https://github.com/tensorflow/tensorflow/issues/24496
  # os.environ["CUDNN_LOGINFO_DBG"] = "1"
  # os.environ["CUDNN_LOGDEST_DBG"] = "stdout"
  # The following might fix (workaround): Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
  # (https://github.com/tensorflow/tensorflow/issues/24496).
  # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

  import tensorflow as tf
  print("TensorFlow:", tf.__version__)

  import returnn
  print("RETURNN:", returnn.__version__)

  import returnn.util.basic as util
  util.init_thread_join_hack()

  from returnn.util import better_exchook
  better_exchook.install()
  better_exchook.replace_traceback_format_tb()

  from returnn.log import log
  log.initialize(verbosity=[5])

  import returnn.tf.util.basic as tf_util
  tf_util.debug_register_better_repr()

  import returnn.util.debug as debug
  debug.install_lib_sig_segfault()

  try:
    import faulthandler
    # Enable after libSigSegfault, so that we have both,
    # because faulthandler will also call the original sig handler.
    faulthandler.enable()
  except ImportError:
    print("no faulthandler")

  # ----------------
  # So far only external deps.
  # Now setup the paths for this repo and do some basic checks.

  if parent_root_dir not in sys.path:
    sys.path.insert(0, parent_root_dir)  # make `import returnn_common` work

  import importlib

  if _expected_mod_name != _my_old_mod_name:
    globals()["__package__"] = _expected_mod_name[:_expected_mod_name.rfind(".")]
    globals()["__name__"] = _expected_mod_name
    sys.modules[_expected_mod_name] = sys.modules[_my_old_mod_name]
    mod = importlib.import_module(_expected_mod_name)
    assert vars(mod) is globals()

  import returnn_common  # noqa

  main_mod = sys.modules.get("__main__")
  if main_mod and os.path.dirname(os.path.realpath(os.path.abspath(main_mod.__file__))) == my_dir:
    main_mod_ = importlib.import_module(
      _expected_package_name + "." + os.path.splitext(os.path.basename(main_mod.__file__))[0])
    _main(main_mod_)


def _main(mod):
  print("Test module:", mod)
  sep = "-" * 80 + "\n"

  if len(sys.argv) > 1:
    tests = sys.argv[1:]
    print("Running functions:", tests)
    print(sep)
    for name in tests:
      print(f"{name}()")
      test = getattr(mod, name)
      test()
      print("Ok.")
      print(sep)
    print("All ok.")
    return

  print("Running tests.")
  print(sep)
  for key, value in vars(mod).items():
    if key.startswith("test_") and callable(value):
      print(f"{key}()")
      value()
      print("Ok.")
      print(sep)
  print("All ok.")


setup()
