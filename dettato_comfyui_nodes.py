"""ComfyUI nodes for audio-to-MIDI transcription with Dettato."""

import json
from io import BytesIO
from pathlib import Path


TRANSCRIPTION_TYPE = "DETTATO_TRANSCRIPTION"


class DettatoTranscribe:
    CATEGORY = "audio/Dettato"
    FUNCTION = "transcribe"
    RETURN_TYPES = (TRANSCRIPTION_TYPE, "STRING")
    RETURN_NAMES = ("transcription", "summary")
    OUTPUT_IS_LIST = (True, True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "device": (["auto", "cpu"], {"default": "auto"}),
            },
        }

    def transcribe(self, audio, device):
        import torch
        from comfy import model_management
        from comfy.utils import ProgressBar
        from dettato import Dettato

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim != 3 or waveform.numel() == 0 or sample_rate <= 0:
            raise ValueError("Dettato needs nonempty AUDIO with shape [batch, channels, samples] and a positive sample rate.")

        device = model_management.get_torch_device() if device == "auto" else torch.device("cpu")
        # Dettato's accelerated inference path uses CUDA kernels and bfloat16.
        if device.type != "cuda":
            device = torch.device("cpu")

        transcription = []
        progress = ProgressBar(waveform.shape[0])
        model = Dettato.from_pretrained(str(device))
        try:
            for clip in waveform:
                model_management.throw_exception_if_processing_interrupted()
                mono = clip.detach().to(device="cpu", dtype=torch.float32).mean(dim=0).numpy()
                transcription.append(model.transcribe(mono, rate=sample_rate))
                progress.update(1)
        finally:
            # Keep the model for this batch only so ComfyUI can reuse the VRAM.
            del model

        return (transcription, [json.dumps(result.summary(), indent=2) for result in transcription])


class DettatoSaveMIDI:
    CATEGORY = "audio/Dettato"
    FUNCTION = "save"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("midi_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcription": (TRANSCRIPTION_TYPE,),
                "filename_prefix": ("STRING", {"default": "Dettato/transcription"}),
            },
        }

    def save(self, transcription, filename_prefix):
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        folder, stem, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
        folder = Path(folder)
        if not folder.resolve().is_relative_to(Path(output_dir).resolve()):
            raise ValueError("MIDI files must be saved inside ComfyUI's output directory.")

        # Encode before creating the destination and reserve each numbered name exclusively.
        midi = BytesIO()
        transcription.to_midi().write(midi)
        while True:
            filename = f"{stem}_{counter:05d}.mid"
            path = folder / filename
            try:
                output = path.open("xb")
            except FileExistsError:
                counter += 1
                continue

            try:
                with output:
                    output.write(midi.getbuffer())
            except OSError:
                path.unlink(missing_ok=True)
                raise
            return ((Path(subfolder) / filename).as_posix(),)


NODE_CLASS_MAPPINGS = {
    "DettatoTranscribe": DettatoTranscribe,
    "DettatoSaveMIDI": DettatoSaveMIDI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DettatoTranscribe": "Dettato Transcribe",
    "DettatoSaveMIDI": "Save MIDI (Dettato)",
}
