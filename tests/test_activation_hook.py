"""Tests for ActivationHook."""

import torch
import torch.nn as nn

from src.hooks import ActivationHook


def test_activation_hook_basic():
    """Test basic hook attachment and activation capture."""
    # Create a simple model with layers
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = SimpleModel()
    hook = ActivationHook(model, layer_indices=[0, 1, 2])
    hook.attach("mlp")

    # Forward pass
    x = torch.randn(2, 10)  # batch_size=2, d=10
    output = model(x)

    # Collect activations
    activations = hook.collect()

    # Verify
    assert len(activations) == 3
    for i, layer_idx in enumerate([0, 1, 2]):
        assert layer_idx in activations
        # Should be (batch=2, 10) since we hook Linear layers
        assert activations[layer_idx].shape == (2, 10)

    hook.detach()


def test_activation_hook_context_manager():
    """Test hook context manager."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = SimpleModel()

    with ActivationHook(model, [0, 1]) as hook:
        hook.attach()
        x = torch.randn(1, 10)
        output = model(x)
        acts = hook.collect()
        assert len(acts) == 2

    # After exiting context, hooks should be detached
    assert len(hook._handles) == 0


def test_topk_sae_basic():
    """Test TopK SAE forward pass."""
    from src.sae import TopKSAE
    from src.sae.topk_sae import SAEConfig

    config = SAEConfig(d_in=64, d_sae=256, k=16, dtype=torch.float32)
    sae = TopKSAE(config)

    # Forward pass
    x = torch.randn(32, 64)  # batch_size=32, d_in=64
    loss, metrics = sae(x)

    # Verify
    assert loss.item() > 0
    assert "mse" in metrics
    assert "explained_variance" in metrics
    assert "sparsity" in metrics
    assert "dead_features" in metrics

    # Sparsity should be close to k / d_sae
    expected_sparsity = config.k / config.d_sae
    assert abs(metrics["sparsity"] - expected_sparsity) < 0.1


def test_activation_buffer():
    """Test activation buffer save/load."""
    from src.data.activation_buffer import ActivationBuffer, load_activations_from_dir
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create buffer
        buffer = ActivationBuffer(
            output_dir=tmpdir,
            layer_indices=[0, 1, 2],
            batch_size=2,
            dtype="float16",
        )

        # Add some batches
        for _ in range(3):
            batch = {
                0: torch.randn(10, 64),
                1: torch.randn(10, 64),
                2: torch.randn(10, 64),
            }
            buffer.add_batch(batch)

        # Flush
        buffer.flush()

        # Verify files were created
        files = list(os.listdir(tmpdir))
        assert len(files) > 0

        # Load back
        activations = load_activations_from_dir(tmpdir)
        assert len(activations) == 3
        for i in [0, 1, 2]:
            assert i in activations
            assert activations[i].shape[0] == 30  # 3 batches * 10 vectors


if __name__ == "__main__":
    test_activation_hook_basic()
    test_activation_hook_context_manager()
    test_topk_sae_basic()
    test_activation_buffer()
    print("All tests passed!")
