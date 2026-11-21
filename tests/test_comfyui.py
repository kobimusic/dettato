import json
import sys
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pretty_midi
import torch

def test_comfyui_batch_transcribes_and_saves_distinct_midi_files(monkeypatch, tmp_path):
    loaded_devices, received_audio, progress = [], [], []

    def transcribe(samples, rate):
        received_audio.append((samples, rate))
        pitch = 60 + len(received_audio)
        piano = dettato.Track("Piano", 0, np.array([[0.0, 0.5, pitch, 80]]))
        drums = dettato.Track("Drums", 0, np.array([[0.1, 0.2, 36, 100]]), is_drum=True)
        return dettato.Transcription("<array>", 0.5, [piano, drums])

    def load_model(device):
        loaded_devices.append(device)
        return SimpleNamespace(transcribe=transcribe)

    def save_path(prefix, output_dir):
        path = Path(prefix)
        folder = Path(output_dir) / path.parent
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder), path.name, 1, str(path.parent), prefix

    comfy = ModuleType("comfy")
    comfy.model_management = SimpleNamespace(
        get_torch_device=lambda: torch.device("cpu"),
        throw_exception_if_processing_interrupted=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", SimpleNamespace(
        ProgressBar=lambda total: SimpleNamespace(update=progress.append),
    ))
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(
        get_output_directory=lambda: str(tmp_path),
        get_save_image_path=save_path,
    ))
    path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = spec_from_file_location("test_comfyui_dettato_extension", path)
    extension = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, extension)
    spec.loader.exec_module(extension)
    dettato = import_module(f"{spec.name}.src.dettato")
    assert Path(dettato.__file__).resolve() == path.parent / "src" / "dettato" / "__init__.py"
    monkeypatch.setattr(dettato.Dettato, "from_pretrained", staticmethod(load_model))
    monkeypatch.setitem(sys.modules, "pretty_midi", None)
    assert (path.parent / extension.WEB_DIRECTORY / "dettato_midi.js").is_file()
    transcriber = extension.NODE_CLASS_MAPPINGS["DettatoTranscribe"]()
    saver = extension.NODE_CLASS_MAPPINGS["DettatoSaveMIDI"]()
    assert transcriber.OUTPUT_IS_LIST == (True, True)
    assert transcriber.RETURN_TYPES[0] == saver.INPUT_TYPES()["required"]["transcription"][0]
    assert saver.OUTPUT_NODE

    waveform = torch.tensor([[[0.0, -1.0], [1.0, 0.0]], [[0.5, 0.5], [0.5, 1.0]]], dtype=torch.float16)
    results, summaries = transcriber.transcribe({"waveform": waveform, "sample_rate": 48000}, "auto")

    assert loaded_devices == ["cpu"]
    assert progress == [1, 1]
    assert [rate for _, rate in received_audio] == [48000, 48000]
    assert all(samples.dtype == np.float32 for samples, _ in received_audio)
    np.testing.assert_array_equal([samples for samples, _ in received_audio], [[0.5, -0.5], [0.5, 0.75]])
    assert [json.loads(summary)["tracks"][0]["notes"] for summary in summaries] == [1, 1]

    exports = [saver.save(result, "Dettato/clip") for result in results]
    paths = [export["result"][0] for export in exports]
    assert paths == ["Dettato/clip_00001.mid", "Dettato/clip_00002.mid"]
    assert [export["ui"]["midi"] for export in exports] == [
        [{"filename": Path(path).name, "subfolder": "Dettato", "type": "output"}] for path in paths
    ]
    for path, pitch in zip(paths, [61, 62]):
        midi = pretty_midi.PrettyMIDI(str(tmp_path / path))
        assert midi.instruments[0].notes[0].pitch == pitch
        assert midi.instruments[1].is_drum
        assert midi.instruments[1].notes[0].pitch == 36


def test_resample_without_soxr(monkeypatch):
    from dettato.audio import resample

    monkeypatch.setitem(sys.modules, "soxr", None)
    samples = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100).astype(np.float32)
    output = resample(samples, 44100, 16000)
    expected = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)
    assert output.dtype == np.float32
    assert output.shape == expected.shape
    np.testing.assert_allclose(output[20:-20], expected[20:-20], atol=0.002)
