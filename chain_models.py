import torch
import torch.nn as nn
import numpy as np

try:
    import onnx
    from onnx import compose
    import onnxruntime as ort
except Exception as e:  # pragma: no cover - handles missing deps gracefully
    raise ImportError('onnx and onnxruntime are required to run this script') from e


torch.manual_seed(0)


class ModelA(nn.Module):
    """A simple 1D convolutional model."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d produces [batch, channels, seq]; transformer expects [seq, batch, embed]
        x = self.conv(x)
        return x.permute(2, 0, 1)


class ModelB(nn.Module):
    """A minimal Transformer model with two inputs."""

    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=4, nhead=2, num_encoder_layers=1, num_decoder_layers=1
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.transformer(src, tgt)


def export_models(a: nn.Module, b: nn.Module, x: torch.Tensor, tgt: torch.Tensor) -> None:
    """Export both models to ONNX files."""
    torch.onnx.export(
        a,
        x,
        "model_a.onnx",
        input_names=["input"],
        output_names=["a_out"],
        opset_version=17,
    )
    dummy_src = torch.randn_like(tgt)
    torch.onnx.export(
        b,
        (dummy_src, tgt),
        "model_b.onnx",
        input_names=["src", "tgt"],
        output_names=["b_out"],
        opset_version=17,
    )


def merge_models() -> None:
    """Merge exported ONNX graphs into a single model."""
    model_a = onnx.load("model_a.onnx")
    model_b = onnx.load("model_b.onnx")
    merged = compose.merge_models(model_a, model_b, io_map=[("a_out", "src")])
    onnx.checker.check_model(merged)
    onnx.save(merged, "chained_model.onnx")


def verify(x: torch.Tensor, tgt: torch.Tensor) -> None:
    """Verify merged ONNX model using ONNX Runtime before and after merge."""
    # Run the two models separately with ONNX Runtime
    sess_a = ort.InferenceSession("model_a.onnx")
    a_out = sess_a.run(None, {"input": x.numpy()})[0]

    sess_b = ort.InferenceSession("model_b.onnx")
    b_out = sess_b.run(None, {"src": a_out, "tgt": tgt.numpy()})[0]

    # Run the merged model
    sess_merged = ort.InferenceSession("chained_model.onnx")
    merged_out = sess_merged.run(None, {"input": x.numpy(), "tgt": tgt.numpy()})[0]

    max_diff = np.max(np.abs(b_out - merged_out))
    print("Max difference between separate and merged ONNX outputs:", max_diff)


if __name__ == "__main__":
    # Sample inputs
    x = torch.randn(1, 3, 10)
    tgt = torch.randn(10, 1, 4)

    # Instantiate models
    model_a = ModelA()
    model_b = ModelB()

    export_models(model_a, model_b, x, tgt)
    merge_models()
    verify(x, tgt)
