"""
Custom callback to log total learnable parameters to WandB metadata
"""

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only


class ParameterCountLogger(pl.Callback):
    """
    Callback to log total number of learnable parameters to WandB metadata
    """

    def __init__(self):
        super().__init__()

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Log parameter count at the start of training"""
        # Count total parameters
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )

        # Get parameter breakdown by module
        param_breakdown = {}
        for name, module in pl_module.named_modules():
            if len(list(module.parameters())) > 0:
                module_params = sum(p.numel() for p in module.parameters())
                param_breakdown[f"params/{name}"] = module_params

        # Log to WandB via the logger
        if hasattr(trainer.logger, "experiment") and hasattr(
            trainer.logger.experiment, "config"
        ):
            # Log to WandB config (metadata)
            try:
                trainer.logger.experiment.config.update(
                    {
                        "model/total_parameters": total_params,
                        "model/trainable_parameters": trainable_params,
                        "model/non_trainable_parameters": total_params
                        - trainable_params,
                        **param_breakdown,
                    }
                )

                # Also log as summary metrics for easy access
                trainer.logger.experiment.summary.update(
                    {
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                    }
                )
                print("✅ Model parameters logged to W&B successfully!")
            except Exception as e:
                print(f"❌ Error logging parameters to W&B: {e}")
                import traceback

                traceback.print_exc()

        print(f"📊 Model Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Non-trainable: {total_params - trainable_params:,}")
