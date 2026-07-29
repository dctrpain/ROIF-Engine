from roif_engine.core.node import Node
from roif_engine.core.distance_constraint import DistanceConstraint


def main() -> None:
    node_a = Node(
        node_id=0,
        position=[0.0, 0.0, 0.0],
        mass=1.0,
        fixed=True,
    )

    node_b = Node(
        node_id=1,
        position=[2.0, 0.0, 0.0],
        mass=1.0,
        fixed=False,
    )

    constraint = DistanceConstraint(
        constraint_id=0,
        node_a=node_a,
        node_b=node_b,
        target_length=1.0,
        compliance=0.0,
    )

    print("Before:")
    print("Node B:", node_b.position)
    print("Violation:", constraint.evaluate())

    constraint.begin_step()
    constraint.project(dt=0.01)
    constraint.end_step()

    print("\nAfter:")
    print("Node B:", node_b.position)
    print("Violation:", constraint.evaluate())


if __name__ == "__main__":
    main()