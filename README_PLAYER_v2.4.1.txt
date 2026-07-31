ROIF Recording Player v2.4.1
============================

Files:

    play.py
    examples/recording_player.py
    tests/test_recording_player.py

Basic playback:

    python play.py output\example_19_recording.roifrec

Alternative launcher:

    python examples\recording_player.py output\example_19_recording.roifrec

Inspect only:

    python play.py output\example_19_recording.roifrec --inspect

Playback modes:

    python play.py recording.roifrec --mode geometry
    python play.py recording.roifrec --mode force
    python play.py recording.roifrec --mode strain
    python play.py recording.roifrec --mode stress
    python play.py recording.roifrec --mode damage

Speed:

    python play.py recording.roifrec --interval 15
    python play.py recording.roifrec --interval 100

Repeat:

    python play.py recording.roifrec --repeat

First 100 frames only:

    python play.py recording.roifrec --frames 100

Export GIF:

    python play.py recording.roifrec --gif output\recording.gif

Hide labels/reference geometry:

    python play.py recording.roifrec --no-labels --no-reference

Help:

    python play.py --help

Tests:

    python -m pytest
