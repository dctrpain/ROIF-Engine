from pathlib import Path
import statistics

import numpy as np

from core.surface_contact import SurfaceContactCandidate
from core.surface_mesh import SurfaceMesh


BODY_PARTS_ROOT = Path(
    r"C:\Users\DELL\OneDrive\Документы\roif-dev\body_parts_3d_api\meshes"
)

TIBIA_FJ = "FJ3387"
TALUS_FJ = "FJ3385"
MAX_DISTANCE_MM = 5.0


def find_mesh_by_fj(fj_id: str) -> Path:
    matches = list(BODY_PARTS_ROOT.glob(f"{fj_id}_*.obj"))

    if len(matches) != 1:
        raise RuntimeError(
            f"{fj_id}: expected exactly one OBJ, found {len(matches)}"
        )

    return matches[0]


def main() -> None:
    tibia_path = find_mesh_by_fj(TIBIA_FJ)
    talus_path = find_mesh_by_fj(TALUS_FJ)

    print(f"tibia: {tibia_path.name}")
    print(f"talus: {talus_path.name}")

    tibia = SurfaceMesh.from_obj(tibia_path)
    talus = SurfaceMesh.from_obj(talus_path)

    print(
        f"triangles: tibia={tibia.triangle_count}, "
        f"talus={talus.triangle_count}"
    )

    candidates = SurfaceContactCandidate.from_nearby_meshes(
        tibia,
        talus,
        max_distance=MAX_DISTANCE_MM,
    )

    facing = [
        candidate
        for candidate in candidates
        if candidate.mutually_facing
    ]

    distances = [
        candidate.distance
        for candidate in candidates
    ]

    facing_distances = [
        candidate.distance
        for candidate in facing
    ]

    print(f"max_distance_mm: {MAX_DISTANCE_MM}")
    print(f"nearby_candidates: {len(candidates)}")
    print(f"facing_candidates: {len(facing)}")

    if candidates:
        print(f"nearby_min_distance_mm: {min(distances):.9f}")
        print(
            "nearby_median_distance_mm: "
            f"{statistics.median(distances):.9f}"
        )
        print(
            "facing_fraction: "
            f"{len(facing) / len(candidates):.6f}"
        )

    if facing:
        print(
            "facing_min_distance_mm: "
            f"{min(facing_distances):.9f}"
        )
        print(
            "facing_median_distance_mm: "
            f"{statistics.median(facing_distances):.9f}"
        )


    if facing:
        unique_tibia = {
            candidate.triangle_a
            for candidate in facing
        }
        unique_talus = {
            candidate.triangle_b
            for candidate in facing
        }

        points_tibia = np.array(
            [candidate.point_a for candidate in facing]
        )
        points_talus = np.array(
            [candidate.point_b for candidate in facing]
        )

        print(f"facing_unique_tibia_triangles: {len(unique_tibia)}")
        print(f"facing_unique_talus_triangles: {len(unique_talus)}")
        print(
            "facing_tibia_bbox_min: "
            f"{points_tibia.min(axis=0).tolist()}"
        )
        print(
            "facing_tibia_bbox_max: "
            f"{points_tibia.max(axis=0).tolist()}"
        )
        print(
            "facing_talus_bbox_min: "
            f"{points_talus.min(axis=0).tolist()}"
        )
        print(
            "facing_talus_bbox_max: "
            f"{points_talus.max(axis=0).tolist()}"
        )

if __name__ == "__main__":
    main()

