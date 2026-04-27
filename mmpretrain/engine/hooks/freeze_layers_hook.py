# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class FreezeLayersHook(Hook):
    """Freeze specified layers during training and optionally unfreeze after
    certain epochs.

    This hook freezes layers whose names match the ``freeze_layers`` patterns
    during the initial training phase, then unfreezes them after a specified
    epoch to allow fine-tuning.

    Args:
        freeze_layers (str | list[str]): Layer name patterns to freeze.
            Can be a string (single pattern) or a list of strings.
            Wildcard ``*`` matches any character sequence.
            Example: ``'backbone'`` freezes all layers whose name contains
            ``'backbone'``. ``['backbone', 'layer1']`` freezes layers
            matching either pattern.
        freeze_epochs (int): Number of epochs to keep layers frozen.
            After this, layers will be unfrozen. Default: 5.
        unfreeze_backbone (bool): Whether to unfreeze backbone after
            freeze_epochs. If True, backbone layers will be unfrozen.
            Default: True.
        priority (str): Hook priority in
            ``PrioritySort`` or `` PRIORITY ``. Lower value means
            higher priority. Default: 'NORMAL'.
    """

    def __init__(self,
                 freeze_layers,
                 freeze_epochs=5,
                 unfreeze_backbone=True,
                 priority='NORMAL'):
        self.freeze_layers = freeze_layers if isinstance(
            freeze_layers, list) else [freeze_layers]
        self.freeze_epochs = freeze_epochs
        self.unfreeze_backbone = unfreeze_backbone
        self.priority = priority

    def _match_layer(self, name, patterns):
        for p in patterns:
            if p == '*':
                return True
            if '*' in p:
                import fnmatch
                return fnmatch.fnmatch(name, p)
            return p in name
        return False

    def _freeze_layers(self, model):
        count = 0
        for name, param in model.named_parameters():
            if self._match_layer(name, self.freeze_layers):
                param.requires_grad = False
                count += 1
        for name, module in model.named_modules():
            if self._match_layer(name, self.freeze_layers):
                module.eval()
        return count

    def _unfreeze_layers(self, model):
        count = 0
        for name, param in model.named_parameters():
            if self._match_layer(name, self.freeze_layers):
                param.requires_grad = True
                count += 1
        return count

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        if epoch == 0:
            frozen = self._freeze_layers(model)
            runner.logger.info(
                f'FreezeLayersHook: Frozen {frozen} parameters '
                f'matching {self.freeze_layers} for epoch 0.')

        if epoch == self.freeze_epochs and self.unfreeze_backbone:
            unfrozen = self._unfreeze_layers(model)
            runner.logger.info(
                f'FreezeLayersHook: Unfrozen {unfrozen} parameters '
                f'at epoch {epoch}. Now training all parameters.')
