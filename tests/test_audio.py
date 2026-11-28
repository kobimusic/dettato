import subprocess

import numpy as np
import pytest
import soundfile as sf

from dettato import audio


def not_audio(tmp_path):
    p = tmp_path / "a.m4a"
    p.write_text("not audio")
    return str(p)


def test_load_reads_a_stereo_wav_as_mono(tmp_path):
    x = np.stack([np.full(800, 0.5, np.float32), np.zeros(800, np.float32)], 1)
    sf.write(tmp_path / "a.wav", x, 8000, subtype="FLOAT")
    a = audio.load(str(tmp_path / "a.wav"))
    assert a.rate == 8000 and a.samples.shape == (800,) and a.samples.dtype == np.float32
    assert np.allclose(a.samples, 0.25)


def test_load_says_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file: .*nope.mp3"):
        audio.load(str(tmp_path / "nope.mp3"))


def test_load_names_ffmpeg_when_it_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(audio.AudioError, match="ffmpeg is not installed") as e:
        audio.load(not_audio(tmp_path))
    assert "a.m4a" in str(e.value) and "soundfile: " in str(e.value)


def test_load_shows_what_ffmpeg_said(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(audio.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 183, b"", b"moov atom not found\nError opening input\n"))
    with pytest.raises(audio.AudioError) as e:
        audio.load(not_audio(tmp_path))
    msg = str(e.value)
    assert "a.m4a: not readable as audio" in msg and "soundfile: " in msg
    assert "ffmpeg: moov atom not found" in msg and "Error opening input" in msg


@pytest.mark.skipif(audio.shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_load_with_the_real_ffmpeg_on_a_file_that_is_not_audio(tmp_path):
    with pytest.raises(audio.AudioError, match="ffmpeg: "):
        audio.load(not_audio(tmp_path))
