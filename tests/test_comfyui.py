import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pretty_midi
import torch

from dettato import Dettato, Track, Transcription


def test_comfyui_batch_transcribes_and_saves_distinct_midi_files(monkeypatch, tmp_path):
    loaded_devices, received_audio, progress = [], [], []

    def transcribe(samples, rate):
        received_audio.append((samples, rate))
        pitch = 60 + len(received_audio)
        track = Track("Piano", 0, np.array([[0.0, 0.5, pitch, 80]]))
        return Transcription("<array>", 0.5, [track])

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
    monkeypatch.setattr(Dettato, "from_pretrained", staticmethod(load_model))

    path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = spec_from_file_location("test_comfyui_dettato_extension", path)
    extension = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, extension)
    spec.loader.exec_module(extension)
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

    paths = [saver.save(result, "Dettato/clip")[0] for result in results]
    assert paths == ["Dettato/clip_00001.mid", "Dettato/clip_00002.mid"]
    for path, pitch in zip(paths, [61, 62]):
        midi = pretty_midi.PrettyMIDI(str(tmp_path / path))
        assert midi.instruments[0].notes[0].pitch == pitch
