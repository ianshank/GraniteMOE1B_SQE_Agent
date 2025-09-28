import pytest

from src.models.granite_moe import GraniteMoETrainer


class _DummyDataset:
    def save_to_disk(self, *_args, **_kwargs):  # pragma: no cover - not expected to run
        return None


def test_prepare_offline_fine_tuning_missing_peft(monkeypatch):
    """prepare_offline_fine_tuning should raise helpful ImportError if peft is missing."""
    # Ensure peft import fails reliably
    monkeypatch.setitem(__import__("sys").modules, "peft", None)
    trainer = GraniteMoETrainer()

    with pytest.raises(ImportError) as exc:
        trainer.prepare_offline_fine_tuning(_DummyDataset())
    assert "peft" in str(exc.value)

