from __future__ import annotations

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - exercised only in minimal sandboxes
    import math, pickle
    class ndarray(list):
        @property
        def shape(self): return (len(self),)
        def astype(self, _): return ndarray(float(x) for x in self)
        def __truediv__(self, n): return ndarray(float(x)/n for x in self)
        def reshape(self, *args): return self
    class _FallbackNumpy:
        uint8 = int; float32 = float; ndarray = ndarray
        @staticmethod
        def array(values, dtype=None): return ndarray(float(x) if dtype is float else x for x in values)
        @staticmethod
        def asarray(values, dtype=None):
            flat=[]
            def walk(v):
                if isinstance(v,(list,tuple)):
                    for x in v: walk(x)
                else: flat.append(v)
            walk(values); return _FallbackNumpy.array(flat, dtype)
        @staticmethod
        def frombuffer(buf, dtype=None): return ndarray(buf)
        @staticmethod
        def zeros(n, dtype=None): return ndarray([0.0]*n)
        @staticmethod
        def dot(a,b): return sum(float(x)*float(y) for x,y in zip(a,b))
        class linalg:
            @staticmethod
            def norm(v): return math.sqrt(sum(float(x)*float(x) for x in v))
        @staticmethod
        def save(path, arr):
            with open(path, 'wb') as f: pickle.dump(list(arr), f)
        @staticmethod
        def load(path):
            with open(path, 'rb') as f: return ndarray(pickle.load(f))
    np = _FallbackNumpy()
