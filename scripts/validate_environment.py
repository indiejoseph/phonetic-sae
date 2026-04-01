#!/usr/bin/env python3
"""Validate the PhoneticSAE environment and dependencies.

Run this before starting Phase 1 to ensure everything is set up correctly.

Usage:
    python scripts/validate_environment.py
    python scripts/validate_environment.py --verbose
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Tuple, List


class EnvironmentValidator:
    """Check that all required dependencies and configurations are in place."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks = []
        self.passed = 0
        self.failed = 0

    def log(self, message: str, level: str = "INFO"):
        """Print a log message."""
        prefix = "ℹ️ " if level == "INFO" else "✅ " if level == "OK" else "⚠️ " if level == "WARN" else "❌ "
        print(f"{prefix} {message}")

    def check(self, name: str, test_fn, critical: bool = True) -> bool:
        """Run a single check."""
        try:
            result, message = test_fn()
            if result:
                self.log(f"{name}: {message}", "OK")
                self.passed += 1
                return True
            else:
                level = "WARN" if not critical else None
                self.log(f"{name}: {message}", "WARN" if not critical else "ERROR")
                self.failed += 1 if critical else 0
                return False
        except Exception as e:
            level = "WARN" if not critical else "ERROR"
            self.log(f"{name}: {str(e)[:80]}", level)
            self.failed += 1 if critical else 0
            return False

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("\n" + "=" * 70)
        print("PHONETIC SAE ENVIRONMENT VALIDATION")
        print("=" * 70 + "\n")

        # Python version
        self.check(
            "Python version",
            lambda: (
                sys.version_info >= (3, 10),
                f"Python {sys.version_info.major}.{sys.version_info.minor} (need 3.10+)"
            ),
            critical=True
        )

        # Repository structure
        self.check(
            "Repository structure",
            self._check_repo_structure,
            critical=True
        )

        # Git submodules
        self.check(
            "Git submodules",
            self._check_git_submodules,
            critical=True
        )

        # PyTorch
        self.check(
            "PyTorch installation",
            self._check_pytorch,
            critical=True
        )

        # CUDA/GPU
        self.check(
            "CUDA availability",
            self._check_cuda,
            critical=False  # Can work on CPU
        )

        # Transformers
        self.check(
            "Transformers library",
            self._check_transformers,
            critical=True
        )

        # TorchAudio
        self.check(
            "TorchAudio installation",
            self._check_torchaudio,
            critical=True
        )

        # Qwen-ASR (for forced alignment)
        self.check(
            "Qwen-ASR installation",
            self._check_qwen_asr,
            critical=True
        )

        # NumPy
        self.check(
            "NumPy installation",
            self._check_numpy,
            critical=True
        )

        # Core imports
        self.check(
            "Core module imports",
            self._check_core_imports,
            critical=True
        )

        # Disk space
        self.check(
            "Disk space (recommended 500GB)",
            self._check_disk_space,
            critical=False
        )

        # HuggingFace cache
        self.check(
            "HuggingFace connectivity",
            self._check_hf_connectivity,
            critical=False
        )

        # Print summary
        print("\n" + "=" * 70)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("=" * 70 + "\n")

        if self.failed == 0:
            self.log("✅ All checks passed! Ready for Phase 1.", "OK")
            return True
        else:
            self.log(f"⚠️ {self.failed} critical check(s) failed. See above for details.", "ERROR")
            return False

    # Individual check methods
    def _check_repo_structure(self) -> Tuple[bool, str]:
        """Check repository has expected directories."""
        required_dirs = ["src", "scripts", "docs", "configs"]
        missing = [d for d in required_dirs if not Path(d).is_dir()]

        if missing:
            return False, f"Missing directories: {missing}"
        return True, "All required directories found"

    def _check_git_submodules(self) -> Tuple[bool, str]:
        """Check git submodules are initialized."""
        try:
            result = subprocess.run(
                ["git", "submodule", "status"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return False, "Git not available or not a git repo"

            lines = result.stdout.strip().split("\n")
            if not lines or not lines[0]:
                return False, "No submodules found"

            uninitialized = [l for l in lines if l.startswith("-")]
            if uninitialized:
                return False, f"Uninitialized submodules: {len(uninitialized)}"

            return True, f"All {len(lines)} submodules initialized"
        except Exception as e:
            return False, str(e)

    def _check_pytorch(self) -> Tuple[bool, str]:
        """Check PyTorch is installed and version is sufficient."""
        try:
            import torch
            version = tuple(map(int, torch.__version__.split(".")[:2]))
            if version < (2, 0):
                return False, f"PyTorch {torch.__version__} (need 2.0+, 2.1+ recommended)"
            return True, f"PyTorch {torch.__version__}"
        except ImportError:
            return False, "PyTorch not installed"

    def _check_cuda(self) -> Tuple[bool, str]:
        """Check CUDA is available."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA not available (will use CPU, slower)"

            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"{device_name} ({memory:.1f}GB)"
        except Exception as e:
            return False, str(e)

    def _check_transformers(self) -> Tuple[bool, str]:
        """Check transformers library."""
        try:
            import transformers
            version = transformers.__version__
            return True, f"Transformers {version}"
        except ImportError:
            return False, "Transformers not installed"

    def _check_torchaudio(self) -> Tuple[bool, str]:
        """Check torchaudio library."""
        try:
            import torchaudio
            version = torchaudio.__version__
            return True, f"TorchAudio {version}"
        except ImportError:
            return False, "TorchAudio not installed"

    def _check_qwen_asr(self) -> Tuple[bool, str]:
        """Check qwen_asr library for forced alignment."""
        try:
            from qwen_asr import Qwen3ForcedAligner
            return True, "Qwen-ASR (Qwen3ForcedAligner available)"
        except ImportError:
            return False, "Qwen-ASR not installed. Run: pip install qwen-asr"

    def _check_numpy(self) -> Tuple[bool, str]:
        """Check numpy library."""
        try:
            import numpy
            version = numpy.__version__
            return True, f"NumPy {version}"
        except ImportError:
            return False, "NumPy not installed"

    def _check_core_imports(self) -> Tuple[bool, str]:
        """Check that core modules can be imported."""
        try:
            from src.hooks import ActivationHook
            from src.sae import TopKSAE
            from src.data import CustomDataset
            return True, "src.hooks, src.sae, src.data all import successfully"
        except ImportError as e:
            return False, str(e)

    def _check_disk_space(self) -> Tuple[bool, str]:
        """Check available disk space."""
        try:
            import shutil
            stat = shutil.disk_usage(".")
            available_gb = stat.free / 1e9

            if available_gb < 100:
                return False, f"Only {available_gb:.1f}GB free (need ~500GB for full capture)"
            elif available_gb < 500:
                return True, f"{available_gb:.1f}GB free (enough for testing, but consider expanding for full capture)"
            else:
                return True, f"{available_gb:.1f}GB free"
        except Exception as e:
            return False, str(e)

    def _check_hf_connectivity(self) -> Tuple[bool, str]:
        """Check HuggingFace Hub connectivity."""
        try:
            # Try to get model info without downloading
            from huggingface_hub import model_info

            # Just check that we can reach HF API
            info = model_info("gpt2", timeout=5)
            return True, "HuggingFace Hub reachable"
        except Exception as e:
            if "404" in str(e):
                return True, "HuggingFace Hub reachable (model check passed)"
            return False, f"HuggingFace connectivity issue: {str(e)[:50]}"


def main():
    parser = argparse.ArgumentParser(description="Validate PhoneticSAE environment")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = EnvironmentValidator(verbose=args.verbose)
    success = validator.validate_all()

    if success:
        print("✅ You're ready to start Phase 1!")
        print("\nNext steps:")
        print("  1. Read: docs/PHASE1_QUICKSTART.md")
        print("  2. Run: python scripts/inspect_aligner_api.py --device cuda")
        print("  3. Run: python scripts/inspect_aligner.py --lang en")
        print("  4. Run: python scripts/capture_with_alignment.py ... (see quickstart for details)")
        sys.exit(0)
    else:
        print("❌ Environment validation failed.")
        print("\nFix the issues above, then run this script again.")
        print("\nFor detailed setup instructions, see: docs/PHASE1_QUICKSTART.md")
        sys.exit(1)


if __name__ == "__main__":
    main()
