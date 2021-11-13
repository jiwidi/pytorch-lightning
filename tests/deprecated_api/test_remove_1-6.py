# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test deprecated functionality which will be removed in v1.6.0."""
import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests.helpers import BoringModel


def test_old_transfer_batch_to_device_hook(tmpdir):
    class OldModel(BoringModel):
        def transfer_batch_to_device(self, batch, device):
            return super().transfer_batch_to_device(batch, device, None)

    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=0, max_epochs=1)
    with pytest.deprecated_call(match="old signature will be removed in v1.6"):
        trainer.fit(OldModel())


def test_v1_6_0_is_overridden_model():
    model = BoringModel()
    with pytest.deprecated_call(match="and will be removed in v1.6"):
        assert is_overridden("validation_step", model=model)
    with pytest.deprecated_call(match="and will be removed in v1.6"):
        assert not is_overridden("foo", model=model)


def test_v1_6_0_deprecated_disable_validation():
    trainer = Trainer()
    with pytest.deprecated_call(match="disable_validation` is deprecated in v1.4"):
        _ = trainer.disable_validation


def test_v1_6_0_deprecated_hpc_load(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    trainer.fit(model)
    trainer.checkpoint_connector.hpc_save(tmpdir, trainer.logger)
    checkpoint_path = trainer.checkpoint_connector.get_max_ckpt_path_from_folder(str(tmpdir))
    with pytest.deprecated_call(match=r"`CheckpointConnector.hpc_load\(\)` was deprecated in v1.4"):
        trainer.checkpoint_connector.hpc_load(checkpoint_path)
