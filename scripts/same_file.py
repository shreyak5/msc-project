import pickle

import numpy as np

file1 = "/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/deca_landmark_embedding.npy"
# file2 = "/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/smirk_landmark_embedding.npy"
# file2= "/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/flame_static_embedding_68.pkl"
file2 = "/home/u6ga/sk3925.u6ga/sk3925_project/msc-project/cvthead_landmark_embedding.npy"


def load(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    return np.load(path, allow_pickle=True)


def deep_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_equal(a[k], b[k]) for k in a)
    if hasattr(a, "shape") and hasattr(b, "shape"):
        a_arr = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        b_arr = b.numpy() if hasattr(b, "numpy") else np.asarray(b)
        return a_arr.shape == b_arr.shape and np.array_equal(a_arr, b_arr)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


a = load(file1)
b = load(file2)

if isinstance(a, np.ndarray) and a.shape == () and a.dtype == object:
    a = a.item()
if isinstance(b, np.ndarray) and b.shape == () and b.dtype == object:
    b = b.item()

if deep_equal(a, b):
    print("Identical")
else:
    print("Different")
