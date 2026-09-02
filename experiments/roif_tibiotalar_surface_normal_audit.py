from pathlib import Path
import math

MESH_ROOT = Path(
    r"C:\Users\DELL\OneDrive\Документы\roif-dev\body_parts_3d_api\meshes"
)

PAIRS = (
    ("FJ3387", "FJ3385"),
    ("FJ6479", "FJ6477"),
)

MAX_DISTANCE_MM = 5.0


def sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    n = norm(v)
    if n <= 1e-12:
        return None
    return tuple(x / n for x in v)


def distance(a, b):
    return norm(sub(a, b))


def load_obj(path):
    vertices = []
    triangles = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                vertices.append(tuple(map(float, p[1:4])))

            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    raw = token.split("/")[0]
                    i = int(raw)
                    if i < 0:
                        i = len(vertices) + i
                    else:
                        i -= 1
                    idx.append(i)

                for j in range(1, len(idx) - 1):
                    triangles.append((idx[0], idx[j], idx[j + 1]))

    out = []

    for i, j, k in triangles:
        a, b, c = vertices[i], vertices[j], vertices[k]

        n = normalize(cross(sub(b, a), sub(c, a)))
        if n is None:
            continue

        centroid = tuple(
            (a[d] + b[d] + c[d]) / 3.0
            for d in range(3)
        )

        out.append((centroid, n))

    return vertices, out


def audit(fj_a, fj_b):
    path_a = next(MESH_ROOT.glob(f"{fj_a}_*.obj"))
    path_b = next(MESH_ROOT.glob(f"{fj_b}_*.obj"))

    _, tris_a = load_obj(path_a)
    _, tris_b = load_obj(path_b)

    matches = []

    for centroid_a, normal_a in tris_a:
        best = None

        for centroid_b, normal_b in tris_b:
            d = distance(centroid_a, centroid_b)

            if best is None or d < best[0]:
                best = (d, centroid_b, normal_b)

        d, centroid_b, normal_b = best

        if d <= MAX_DISTANCE_MM:
            normal_dot = dot(normal_a, normal_b)

            direction = normalize(sub(centroid_b, centroid_a))
            facing_a = dot(normal_a, direction) if direction else 0.0
            facing_b = dot(normal_b, tuple(-x for x in direction)) if direction else 0.0

            matches.append(
                (d, normal_dot, facing_a, facing_b)
            )

    print()
    print(f"PAIR {fj_a} <-> {fj_b}")
    print("TIBIA_TRIANGLES =", len(tris_a))
    print("TALUS_TRIANGLES =", len(tris_b))
    print("NEAR_TRIANGLES_WITHIN_5MM =", len(matches))

    if not matches:
        return

    distances = sorted(x[0] for x in matches)
    normal_dots = sorted(x[1] for x in matches)

    opposing = sum(1 for x in matches if x[1] < -0.5)
    strongly_opposing = sum(1 for x in matches if x[1] < -0.8)

    mutually_facing = sum(
        1 for x in matches
        if x[2] > 0 and x[3] > 0
    )

    print("MIN_CENTROID_DISTANCE_MM =", min(distances))
    print("MEDIAN_CENTROID_DISTANCE_MM =", distances[len(distances)//2])
    print("MEDIAN_NORMAL_DOT =", normal_dots[len(normal_dots)//2])

    print("OPPOSING_NORMALS_DOT_LT_-0.5 =", opposing)
    print("STRONGLY_OPPOSING_DOT_LT_-0.8 =", strongly_opposing)
    print("MUTUALLY_FACING =", mutually_facing)

    print(
        "OPPOSING_FRACTION =",
        opposing / len(matches)
    )

    print(
        "STRONGLY_OPPOSING_FRACTION =",
        strongly_opposing / len(matches)
    )

    print(
        "MUTUALLY_FACING_FRACTION =",
        mutually_facing / len(matches)
    )


if __name__ == "__main__":
    for pair in PAIRS:
        audit(*pair)
