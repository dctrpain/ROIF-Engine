from __future__ import annotations

import argparse
from pathlib import Path
import sys

from core.storage import (
    RecordingFileInfo,
    RecordingStorage,
    RecordingStorageError,
)
from visualization import NetworkAnimator


VERSION = "2.4.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="play.py",
        description=(
            "ROIF Recording Player — load and replay "
            ".roifrec simulation recordings."
        ),
    )

    parser.add_argument(
        "recording",
        type=Path,
        help="Path to a .roifrec recording file.",
    )
    parser.add_argument(
        "--mode",
        default="force",
        choices=(
            "geometry",
            "force",
            "strain",
            "stress",
            "damage",
        ),
        help="Visualization mode. Default: force.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        metavar="MS",
        help=(
            "Delay between visual frames in milliseconds. "
            "Default: 30."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "Play only the first N recorded frames. "
            "Default: all frames."
        ),
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Repeat playback continuously.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Hide reference geometry.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide node and element labels.",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help=(
            "Skip SHA-256 verification while loading. "
            "Not recommended."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "Print recording information and exit "
            "without opening the player."
        ),
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        metavar="OUTPUT.gif",
        help=(
            "Export the recording to GIF instead of "
            "opening an interactive window."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="GIF export resolution. Default: 120.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ROIF Recording Player {VERSION}",
    )

    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.interval <= 0.0:
        parser.error("--interval must be positive")

    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")

    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    if args.inspect and args.gif is not None:
        parser.error(
            "--inspect and --gif cannot be used together"
        )

    if (
        args.gif is not None
        and args.gif.suffix.lower() != ".gif"
    ):
        parser.error("--gif output path must end with .gif")


def print_info(info: RecordingFileInfo) -> None:
    print("=" * 88)
    print(
        f"ROIF Recording Player v{VERSION}"
    )
    print("=" * 88)
    print(f"File           : {info.path}")
    print(f"Format         : {info.format_name}")
    print(f"Format version : {info.format_version}")
    print(f"Storage version: {info.storage_version}")
    print(f"Compressed     : {info.compressed}")
    print(f"File size      : {info.file_size_bytes:,} bytes")
    print(f"Frames         : {info.frame_count}")
    print(f"Physical steps : {info.physical_steps}")
    print(f"Simulation dt  : {info.simulation_dt:.9f} s")
    print(f"Sample every   : {info.sample_every}")
    print(f"Start time     : {info.start_time:.6f} s")
    print(f"End time       : {info.end_time:.6f} s")
    print(f"Duration       : {info.duration:.6f} s")
    print(f"SHA-256        : {info.payload_sha256}")

    if info.metadata:
        print("Metadata       :")
        for key, value in info.metadata.items():
            print(f"  {key}: {value}")


def run_player(
    args: argparse.Namespace,
) -> Path | None:
    verify_checksum = not args.no_checksum

    info = RecordingStorage.inspect(
        args.recording,
        verify_checksum=verify_checksum,
    )
    print_info(info)

    if args.inspect:
        print()
        print("Inspection completed.")
        return None

    print()
    print("Loading recording...")

    recording = RecordingStorage.load(
        args.recording,
        verify_checksum=verify_checksum,
    )

    frames = args.frames
    if frames is not None and frames > len(recording):
        raise ValueError(
            "--frames cannot exceed recording length "
            f"({len(recording)})"
        )

    animator = NetworkAnimator.from_recording(
        recording
    )

    if args.gif is not None:
        print(f"Exporting GIF  : {args.gif}")

        output_path = animator.save_gif(
            args.gif,
            frames=frames,
            interval_ms=args.interval,
            mode=args.mode,
            show_reference=not args.no_reference,
            show_labels=not args.no_labels,
            title="ROIF Recording Playback",
            repeat=args.repeat,
            dpi=args.dpi,
        )

        print(f"GIF saved      : {output_path}")
        return output_path

    print()
    print("Opening playback window...")
    print(
        "Close the Matplotlib window to return "
        "to the terminal."
    )

    animator.show(
        frames=frames,
        interval_ms=args.interval,
        mode=args.mode,
        show_reference=not args.no_reference,
        show_labels=not args.no_labels,
        title="ROIF Recording Playback",
        repeat=args.repeat,
        complete_on_close=False,
    )

    print()
    print("Playback closed.")
    return None


def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)

    try:
        run_player(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (
        RecordingStorageError,
        ValueError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
