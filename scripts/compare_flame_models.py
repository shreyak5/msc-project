"""Compare FLAME2020 and FLAME2023 generic_model.pkl files.

Usage:
    python scripts/compare_flame_models.py [flame2020.pkl] [flame2023.pkl]
"""
import sys
import inspect

# --- Compatibility shims so the old chumpy-pickled FLAME models can be
# unpickled under modern numpy/python (chumpy still references numpy
# aliases and an inspect API that were both removed upstream). ---
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import numpy as np

for _name, _val in [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("unicode", str),
    ("str", str),
]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)

import pickle

# DEFAULT_FLAME2020 = "assets/FLAME2020/generic_model.pkl"
# DEFAULT_FLAME2023 = "assets/FLAME2023_Open/flame2023_Open.pkl"
DEFAULT_FLAME2023 = "assets/FLAME2023/flame2023.pkl"


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def to_array(value):
    """chumpy Ch objects, scipy sparse matrices, etc. -> plain ndarray."""
    if hasattr(value, "toarray"):
        return value.toarray()
    if hasattr(value, "r"):  # chumpy Ch
        return np.asarray(value.r)
    return np.asarray(value)


def describe(value):
    try:
        arr = to_array(value)
        return f"{type(value).__name__:<20} shape={arr.shape} dtype={arr.dtype}"
    except Exception:
        return f"{type(value).__name__:<20} value={value!r}"


def compare_arrays(name, a, b):
    try:
        arr_a, arr_b = to_array(a), to_array(b)
    except Exception as e:
        print(f"  [{name}] could not convert to array: {e}")
        return

    if arr_a.shape != arr_b.shape:
        print(f"  [{name}] SHAPE DIFFERS: {arr_a.shape} vs {arr_b.shape}")
        return

    if arr_a.dtype.kind in "fc" or arr_b.dtype.kind in "fc":
        equal = np.allclose(arr_a.astype(float), arr_b.astype(float), equal_nan=True)
    else:
        equal = np.array_equal(arr_a, arr_b)

    status = "identical" if equal else "VALUES DIFFER"
    print(f"  [{name}] shape={arr_a.shape} dtype={arr_a.dtype} -> {status}")


def main():
    path_2020 = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FLAME2020
    path_2023 = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FLAME2023

    print(f"Loading FLAME2020 from {path_2020}")
    m2020 = load(path_2020)
    print(f"Loading FLAME2023 from {path_2023}")
    m2023 = load(path_2023)

    keys_2020, keys_2023 = set(m2020), set(m2023)
    print(f'2020 keys: {keys_2020}')
    print(f'2023 keys: {keys_2023}')

    print("\n=== Keys ===")
    only_2020 = keys_2020 - keys_2023
    only_2023 = keys_2023 - keys_2020
    print(f"Only in FLAME2020: {sorted(only_2020) or 'none'}")
    print(f"Only in FLAME2023: {sorted(only_2023) or 'none'}")

    print("\n=== Per-key comparison (shared keys) ===")
    for key in sorted(keys_2020 & keys_2023):
        v2020, v2023 = m2020[key], m2023[key]
        if isinstance(v2020, (np.ndarray, list)) or hasattr(v2020, "shape") or hasattr(v2020, "toarray"):
            compare_arrays(key, v2020, v2023)
        else:
            equal = v2020 == v2023
            print(f"  [{key}] {v2020!r} vs {v2023!r} -> {'identical' if equal else 'DIFFERS'}")

    print("\n=== Keys unique to one model ===")
    for key in sorted(only_2020):
        print(f"  FLAME2020.{key}: {describe(m2020[key])}")
    for key in sorted(only_2023):
        print(f"  FLAME2023.{key}: {describe(m2023[key])}")
        print(m2023[key])


if __name__ == "__main__":
    main()
