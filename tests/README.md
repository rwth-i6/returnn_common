
You can run tests either by calling them directly
(from the root dir) like this::

    python -m tests.test_hello

Or (from parent root dir)::

    python -m returnn_code.tests.test_hello

You can also pass a specific test function like::

    python -m tests.test_hello test_hello

You can also use nosetests::

    nosetests tests/test_hello.py

---

When you want to create a new test file,
create ``test_<name>.py``,
and add this in the beginning::

    from . import _setup_test_env  # noqa

See ``_setup_test_env`` for some implementation details.

To add the test file to GitHub CI,
edit file ``.github/workflows/main.yml``.
