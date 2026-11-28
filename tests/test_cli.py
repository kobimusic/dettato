import pytest

from dettato import audio, cli, pipeline


def test_a_missing_input_stops_the_run_before_the_models_load(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.Dettato, "from_pretrained", classmethod(lambda cls, device=None: pytest.fail("the models were loaded")))
    there = tmp_path / "there.wav"
    there.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        cli.main([str(there), str(tmp_path / "nope.mp3"), "-o", str(tmp_path / "out")])
    assert "no such file or folder" in str(e.value.code) and "nope.mp3" in str(e.value.code)
    assert "there.wav" not in str(e.value.code)


def test_a_file_that_cannot_be_read_is_reported_with_the_reason(tmp_path, monkeypatch, capsys):
    class Model:                                        # stands in for the four models: only reads the file
        device = "cpu"

        def transcribe(self, path):
            return audio.load(path)

    monkeypatch.setattr(pipeline.Dettato, "from_pretrained", classmethod(lambda cls, device=None: Model()))
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    bad = tmp_path / "a.m4a"
    bad.write_text("not audio")
    with pytest.raises(SystemExit) as e:
        cli.main([str(bad), "-o", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert e.value.code == 1
    assert f"FAILED {bad}: " in out and "ffmpeg is not installed" in out and "AudioError(" not in out
