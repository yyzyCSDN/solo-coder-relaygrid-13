"""Minimal test runner that does not require pytest.

Provides pytest.approx and pytest.raises compatible shims, plus a capsys
fixture, then runs all test_* functions in the given modules.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import sys
import traceback
from collections import namedtuple


class Approx:
    def __init__(self, expected, rel=1e-6, abs_tol=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs_tol = abs_tol

    def __eq__(self, actual):
        tol = max(abs(self.expected) * self.rel, self.abs_tol)
        return abs(actual - self.expected) <= tol

    def __repr__(self):
        return f"approx({self.expected})"


def approx(expected, rel=1e-6, abs_tol=1e-12):
    return Approx(expected, rel, abs_tol)


class Raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.exc_type.__name__} but no exception raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        self.value = exc_val
        return True


def raises(exc_type):
    return Raises(exc_type)


class Capsys:
    """Captures stdout and stderr, mimicking pytest's capsys fixture."""

    def __init__(self):
        self._out = io.StringIO()
        self._err = io.StringIO()
        self._ctx = contextlib.ExitStack()

    def __enter__(self):
        self._ctx.enter_context(contextlib.redirect_stdout(self._out))
        self._ctx.enter_context(contextlib.redirect_stderr(self._err))
        return self

    def __exit__(self, *exc):
        self._ctx.close()
        return False

    def readouterr(self):
        OutErr = namedtuple("OutErr", ["out", "err"])
        out = self._out.getvalue()
        err = self._err.getvalue()
        self._out.seek(0)
        self._out.truncate(0)
        self._err.seek(0)
        self._err.truncate(0)
        return OutErr(out, err)


def run_module(module_name: str) -> tuple[int, int]:
    mod = importlib.import_module(module_name)
    passed = 0
    failed = 0
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        # Inject capsys if the function accepts it.
        import inspect

        sig = inspect.signature(fn)
        kwargs = {}
        if "capsys" in sig.parameters:
            capsys = Capsys()
            with capsys:
                try:
                    fn(capsys)
                    print(f"  PASS {name}")
                    passed += 1
                except Exception:
                    print(f"  FAIL {name}")
                    traceback.print_exc()
                    failed += 1
        else:
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception:
                print(f"  FAIL {name}")
                traceback.print_exc()
                failed += 1
    return passed, failed


def main():
    # Inject shims into pytest namespace before importing tests.
    import types

    pytest_shim = types.ModuleType("pytest")
    pytest_shim.approx = approx
    pytest_shim.raises = raises
    sys.modules["pytest"] = pytest_shim

    modules = sys.argv[1:] or ["tests.test_state_estimation"]
    total_passed = 0
    total_failed = 0
    for mod_name in modules:
        print(f"=== {mod_name} ===")
        p, f = run_module(mod_name)
        total_passed += p
        total_failed += f
        print(f"  ({p} passed, {f} failed)")
    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
