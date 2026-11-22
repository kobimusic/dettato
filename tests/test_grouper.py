"""CPU smoke test of the grouper stage (skipped when the weights cannot be loaded)."""
import numpy as np
import pytest
import torch

from dettato.grouper import INSTRUMENTS, PROGRAM, Grouper
from dettato.grouper import decode as GD
from dettato.grouper import retune
from dettato.grouper.core import assign_slots, song_windows


@pytest.fixture(scope="module")
def stages():
    try:
        from dettato.transcriber import Transcriber

        return Grouper.from_pretrained("cpu"), Transcriber.from_pretrained("cpu")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"grouper / transcriber weights not available: {e!r}")


def tones(seconds=4.0, sr=16000):
    """A low and a high line: two notes each, a little noise."""
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for f0, a, b in ((110.0, 0.0, 2.0), (98.0, 2.0, 4.0), (880.0, 0.5, 1.5), (987.77, 2.5, 3.5)):
        m = (t >= a) & (t < b)
        env = np.exp(-(t[m] - a) * 2.0)
        x[m] += 0.2 * env * sum(np.sin(2 * np.pi * f0 * h * (t[m] - a)) / h for h in (1, 2, 3))
    rng = np.random.default_rng(0)
    return (x + 0.001 * rng.standard_normal(len(x))).astype(np.float32)


def test_vocabulary():
    assert len(INSTRUMENTS) == 73 and len(PROGRAM) == 73
    assert INSTRUMENTS[0] == "piano" and PROGRAM[INSTRUMENTS.index("tenor_sax")] == 66


def test_retune_in_tune_audio_is_left_alone():
    t = np.arange(16000 * 3) / 16000
    x = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    delta, strength, shift, r = retune.ratio(x)
    assert abs(delta) < 5 and shift == 0.0 and r == 1.0


def test_assign_slots():
    notes = np.array([[1.0, 1.01, 60], [0.5, 1.0, 64]])
    own = np.array([[0.48, 1.0, 64, 1.0], [1.3, 1.5, 60, 1.0]])
    out = assign_slots(notes, own)
    assert out.tolist() == [[0.5, 1.0, 64, 1.0], [1.0, 1.02, 60, 0.0]]


def test_song_windows_cover_every_note():
    on = np.sort(np.random.default_rng(1).uniform(0, 100, 500)).astype(np.float32)
    covered = np.zeros(len(on), bool)
    for _, sel in song_windows(on, 24.0, 12.0, 768):
        covered[sel] = True
    assert covered.all()


def test_absorb_small_dissolves_a_stray_track():
    rng = np.random.default_rng(0)
    emb = np.concatenate([rng.normal([1, 0, 0], 0.05, (60, 3)), rng.normal([0.95, 0.1, 0], 0.05, (4, 3))])
    inv = np.array([0] * 60 + [1] * 4)
    on = np.arange(64, dtype=np.float64)
    out = GD.absorb_small(inv, emb, on, on + 0.5, 0.25, 0.3)
    assert (out == 0).all()


def test_group(stages):
    g, t = stages
    notes = np.array([[0.0, 2.0, 45], [2.0, 4.0, 43], [0.5, 1.5, 81], [2.5, 3.5, 83]], np.float64)
    tracks, info = g.group(tones(), notes, t)
    assert info["notes_in"] == 4 and info["notes_out"] == sum(len(tr["notes"]) for tr in tracks)
    assert 1 <= len(tracks) <= 4
    seen = []
    for tr in tracks:
        assert tr["instrument"] in INSTRUMENTS and tr["program"] == PROGRAM[INSTRUMENTS.index(tr["instrument"])]
        n = tr["notes"]
        assert n.ndim == 2 and n.shape[1] == 4
        assert np.all(n[:, 1] >= n[:, 0] + 0.02 - 1e-9) and np.all((n[:, 3] >= 1) & (n[:, 3] <= 127))
        assert np.all(np.diff(n[:, 0]) >= 0)
        seen += n[:, 2].astype(int).tolist()
    assert sorted(seen) == [43, 45, 81, 83]
    assert [len(tr["notes"]) for tr in tracks] == sorted((len(tr["notes"]) for tr in tracks), reverse=True)


def test_group_from_a_shared_pass_equals_group(stages):
    """Dettato.pitched runs the transcriber once; the grouper's features come from its every-other-window average."""
    g, t = stages
    x = tones(12.0)
    notes = np.array([[0.0, 2.0, 45], [2.0, 4.0, 43], [0.5, 1.5, 81], [2.5, 3.5, 83], [5.0, 7.0, 45], [8.0, 11.0, 47]], np.float64)
    ref_tracks, ref_info = g.group(x, notes, t)
    tuning = retune.ratio(x)
    maps, _, (smaps, sbufs) = t.run(x, hop_frac=0.25, taps=("hidden",), sub_hop_frac=0.5, sub_taps=("hidden", "stem"))
    tracks, info = g.group(x, notes, t, tuning=tuning, run=(smaps, sbufs))
    assert info == ref_info
    assert [tr["instrument"] for tr in tracks] == [tr["instrument"] for tr in ref_tracks]
    for a, b in zip(tracks, ref_tracks):
        assert np.array_equal(a["notes"], b["notes"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_group_on_mps_takes_float64_notes(stages):
    """MPS (Apple's Metal) has no float64: the notes the pipeline passes as float64 are moved there as float32."""
    g, t = stages
    try:
        from dettato.transcriber import Transcriber

        gm, tm = Grouper.from_pretrained("mps"), Transcriber.from_pretrained("mps")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"grouper / transcriber weights not available: {e!r}")
    notes = np.array([[0.0, 2.0, 45], [2.0, 4.0, 43], [0.5, 1.5, 81], [2.5, 3.5, 83]], np.float64)
    tracks, info = gm.group(tones(), notes, tm)
    ref_tracks, ref_info = g.group(tones(), notes, t)
    assert info["notes_out"] == ref_info["notes_out"]
    assert sorted(int(p) for tr in tracks for p in tr["notes"][:, 2]) == sorted(int(p) for tr in ref_tracks for p in tr["notes"][:, 2])
