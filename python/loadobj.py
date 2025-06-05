import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable

@dataclass
class ModelObject:
    vertices: List[np.ndarray] = field(default_factory=list)
    uvs: List[np.ndarray] = field(default_factory=list)
    normals: List[np.ndarray] = field(default_factory=list)
    status: bool = False

    def __init__(self, loadpath: str = None):
        self.vertices = []
        self.uvs = []
        self.normals = []
        self.status = False
        if loadpath:
            self.load(loadpath)

    def load(self, loadpath: str):
        try:
            with open(loadpath, 'r') as f:
                temp_vertices = []
                temp_uvs = []
                temp_normals = []
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        temp_vertices.append(np.array([float(p) for p in parts[1:4]], dtype=np.float32))
                    elif line.startswith('vt '):
                        parts = line.strip().split()
                        temp_uvs.append(np.array([float(p) for p in parts[1:3]], dtype=np.float32))
                    elif line.startswith('vn '):
                        parts = line.strip().split()
                        temp_normals.append(np.array([float(p) for p in parts[1:4]], dtype=np.float32))
                    elif line.startswith('f '):
                        parts = line.strip().split()[1:]
                        for part in parts:
                            v_idx, uv_idx, n_idx = (int(x) for x in part.split('/'))
                            self.vertices.append(temp_vertices[v_idx - 1])
                            self.uvs.append(temp_uvs[uv_idx - 1])
                            self.normals.append(temp_normals[n_idx - 1])
            self.status = True
        except OSError:
            print(f"Error occurred when loading {loadpath}")
            self.status = False

    def apply(self, f: Callable[[np.ndarray], np.ndarray]):
        self.vertices = [f(v) for v in self.vertices]

    def multiby_mat3(self, matrix: np.ndarray):
        self.vertices = [matrix @ v for v in self.vertices]
        self.normals = [matrix @ n for n in self.normals]

    def multiby_mat4(self, matrix: np.ndarray):
        def transform(v):
            v4 = np.append(v, 1.0)
            res = matrix @ v4
            return res[:3]
        self.vertices = [transform(v) for v in self.vertices]
        def transform_n(n):
            n4 = np.append(n, 0.0)
            res = matrix @ n4
            return res[:3]
        self.normals = [transform_n(n) for n in self.normals]

    def append(self, other: 'ModelObject'):
        self.vertices.extend(other.vertices)
        self.uvs.extend(other.uvs)
        self.normals.extend(other.normals)


def load_objects_from_txt(path: str):
    objlist: List[ModelObject] = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith('#'):
            filename = lines[i][1:].strip()
            obj = ModelObject(filename)
            i += 1
            while i < len(lines) and lines[i] != 'end':
                if lines[i] == 'operate':
                    mat = []
                    for j in range(4):
                        parts = [float(x) for x in lines[i+1+j].split()]
                        mat.append(parts)
                    mat_np = np.array(mat, dtype=np.float32).T
                    obj.multiby_mat4(mat_np)
                    i += 5
                else:
                    i += 1
            objlist.append(obj)
        i += 1
    return objlist
