import pytest
import os

from torch.optim import SGD

from neuralmagicML.pytorch.recal import (
    GradualKSModifier,
    ConstantKSModifier,
    DimensionSparsityMaskCreator,
    BlockSparsityMaskCreator,
)

from tests.pytorch.helpers import LinearNet
from tests.pytorch.recal.test_modifier import (
    ScheduledModifierTest,
    ScheduledUpdateModifierTest,
    test_epoch,
    test_steps_per_epoch,
    test_loss,
    create_optim_sgd,
    create_optim_adam,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: ConstantKSModifier(layers="__ALL__",),
        lambda: ConstantKSModifier(
            layers=[LinearNet.layer_descs()[2].name], start_epoch=10.0, end_epoch=25.0,
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function",
)
class TestConstantKSModifier(ScheduledModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model, optimizer)

        # check sparsity is not set before
        if modifier.start_epoch >= 0:
            for epoch in range(int(modifier.start_epoch)):
                assert not modifier.update_ready(epoch, test_steps_per_epoch)

        epoch = int(modifier.start_epoch) if modifier.start_epoch >= 0 else 0.0
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

        if modifier.end_epoch >= 0:
            epoch = int(modifier.end_epoch)
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)

            for epoch in range(
                int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6
            ):
                assert not modifier.update_ready(epoch, test_steps_per_epoch)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_constant_ks_yaml():
    layers = "__ALL__"
    start_epoch = 5.0
    end_epoch = 15.0
    param = "weight"
    yaml_str = f"""
    !ConstantKSModifier
        layers: {layers}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        param: {param}
    """
    yaml_modifier = ConstantKSModifier.load_obj(yaml_str)  # type: ConstantKSModifier
    serialized_modifier = ConstantKSModifier.load_obj(
        str(yaml_modifier)
    )  # type: ConstantKSModifier
    obj_modifier = ConstantKSModifier(
        layers=layers, start_epoch=start_epoch, end_epoch=end_epoch, param=param
    )

    assert isinstance(yaml_modifier, ConstantKSModifier)
    assert yaml_modifier.layers == serialized_modifier.layers == obj_modifier.layers
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert yaml_modifier.param == serialized_modifier.param == obj_modifier.param


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
@pytest.mark.parametrize(
    "modifier_lambda",
    [
        lambda: GradualKSModifier(
            layers="__ALL__",
            init_sparsity=0.05,
            final_sparsity=0.95,
            start_epoch=0.0,
            end_epoch=15.0,
            update_frequency=1.0,
            param="weight",
            inter_func="linear",
        ),
        lambda: GradualKSModifier(
            layers=[LinearNet.layer_descs()[2].name],
            init_sparsity=0.05,
            final_sparsity=0.95,
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=1.0,
            inter_func="cubic",
        ),
        lambda: GradualKSModifier(
            layers=[LinearNet.layer_descs()[2].name],
            init_sparsity=0.05,
            final_sparsity=0.95,
            start_epoch=10.0,
            end_epoch=25.0,
            update_frequency=1.0,
            inter_func="cubic",
            mask_type=[1, 4],
        ),
    ],
    scope="function",
)
@pytest.mark.parametrize("model_lambda", [LinearNet], scope="function")
@pytest.mark.parametrize(
    "optim_lambda", [create_optim_sgd, create_optim_adam], scope="function",
)
class TestGradualKSModifier(ScheduledUpdateModifierTest):
    def test_lifecycle(
        self, modifier_lambda, model_lambda, optim_lambda, test_steps_per_epoch
    ):
        modifier = modifier_lambda()
        model = model_lambda()
        optimizer = optim_lambda(model)
        self.initialize_helper(modifier, model, optimizer)
        assert modifier.applied_sparsity is None

        # check sparsity is not set before
        for epoch in range(int(modifier.start_epoch)):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity is None

        epoch = int(modifier.start_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        assert modifier.applied_sparsity == modifier.init_sparsity
        last_sparsity = modifier.init_sparsity

        while epoch < modifier.end_epoch - modifier.update_frequency:
            epoch += modifier.update_frequency
            assert modifier.update_ready(epoch, test_steps_per_epoch)
            modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity > last_sparsity
            last_sparsity = modifier.applied_sparsity

        epoch = int(modifier.end_epoch)
        assert modifier.update_ready(epoch, test_steps_per_epoch)
        modifier.scheduled_update(model, optimizer, epoch, test_steps_per_epoch)
        assert modifier.applied_sparsity == modifier.final_sparsity

        for epoch in range(int(modifier.end_epoch) + 1, int(modifier.end_epoch) + 6):
            assert not modifier.update_ready(epoch, test_steps_per_epoch)
            assert modifier.applied_sparsity == modifier.final_sparsity


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False), reason="Skipping pytorch tests",
)
def test_gradual_ks_yaml():
    layers = "__ALL__"
    init_sparsity = 0.05
    final_sparsity = 0.8
    start_epoch = 5.0
    end_epoch = 15.0
    update_frequency = 1.0
    param = "weight"
    inter_func = "cubic"
    mask_type = 'filter'
    yaml_str = f"""
    !GradualKSModifier
        layers: {layers}
        init_sparsity: {init_sparsity}
        final_sparsity: {final_sparsity}
        start_epoch: {start_epoch}
        end_epoch: {end_epoch}
        update_frequency: {update_frequency}
        param: {param}
        inter_func: {inter_func}
        mask_type: {mask_type}
    """
    yaml_modifier = GradualKSModifier.load_obj(yaml_str)  # type: GradualKSModifier
    serialized_modifier = GradualKSModifier.load_obj(
        str(yaml_modifier)
    )  # type: GradualKSModifier
    obj_modifier = GradualKSModifier(
        layers=layers,
        init_sparsity=init_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        update_frequency=update_frequency,
        param=param,
        inter_func=inter_func,
        mask_type=mask_type,
    )

    assert isinstance(yaml_modifier, GradualKSModifier)
    assert yaml_modifier.layers == serialized_modifier.layers == obj_modifier.layers
    assert (
        yaml_modifier.init_sparsity
        == serialized_modifier.init_sparsity
        == obj_modifier.init_sparsity
    )
    assert (
        yaml_modifier.final_sparsity
        == serialized_modifier.final_sparsity
        == obj_modifier.final_sparsity
    )
    assert (
        yaml_modifier.start_epoch
        == serialized_modifier.start_epoch
        == obj_modifier.start_epoch
    )
    assert (
        yaml_modifier.end_epoch
        == serialized_modifier.end_epoch
        == obj_modifier.end_epoch
    )
    assert (
        yaml_modifier.update_frequency
        == serialized_modifier.update_frequency
        == obj_modifier.update_frequency
    )
    assert yaml_modifier.param == serialized_modifier.param == obj_modifier.param
    assert (
        yaml_modifier.inter_func
        == serialized_modifier.inter_func
        == obj_modifier.inter_func
    )
    assert (
        str(yaml_modifier.mask_type)
        == str(serialized_modifier.mask_type)
        == str(obj_modifier.mask_type)
    )
