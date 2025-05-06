import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch # Updated import
from scipy.spatial import ConvexHull
import numpy as np
import logging # Added for logging

logger = logging.getLogger(__name__)

class BaseGeometricBaseline(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False # Disable automatic optimization

    def forward(self, data: Batch) -> torch.Tensor:
        raise NotImplementedError

    def _common_step(self, batch: Batch, batch_idx: int, step_name: str):
        # Ensure data is a PyG Batch object if it's not already
        # This helps ensure batch.num_graphs and batch.ptr are available
        if not isinstance(batch, Batch):
            # This might happen if batch_size=1 and dataloader yields single Data objects
            logger.debug(f"Converting single Data object to Batch in _common_step for {step_name}")
            batch = Batch.from_data_list([batch])
            
        pred_com = self(batch) 
        true_com = batch.y     

        if true_com.ndim == 3 and true_com.shape[1] == 1:
            true_com = true_com.squeeze(1)
        
        # Debugging shapes if warning persists
        if pred_com.shape[0] != true_com.shape[0] and batch.num_graphs > 1 :
             logger.warning(
                f"Shape mismatch in _common_step ({step_name}): "
                f"pred_com shape: {pred_com.shape}, true_com shape: {true_com.shape}, "
                f"batch.num_graphs: {batch.num_graphs}"
            )

        loss = nn.functional.mse_loss(pred_com, true_com)
        
        # Use batch.num_graphs for logging batch_size, default to 1 if not available
        log_batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') and batch.num_graphs > 0 else 1
        
        self.log(f'{step_name}_loss', loss, batch_size=log_batch_size, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Batch, batch_idx: int):
        # automatic_optimization is False.
        # We just compute and log loss. Lightning won't call .backward() or optimizer.step().
        loss = self._common_step(batch, batch_idx, "train")
        return loss # This loss can be used by loggers or callbacks.

    def validation_step(self, batch: Batch, batch_idx: int):
        self._common_step(batch, batch_idx, "val")
        return None 

    def test_step(self, batch: Batch, batch_idx: int):
        self._common_step(batch, batch_idx, "test")
        return None

    def configure_optimizers(self):
        return None

class CentroidBaseline(BaseGeometricBaseline):
    def forward(self, data: Batch) -> torch.Tensor:
        if not hasattr(data, 'batch') or data.batch is None:
            # Handle single graph case (not batched by PyG DataLoader)
            # This can happen if a Data object is passed directly, not a Batch object
            logger.debug("CentroidBaseline: data.batch not found, assuming single graph.")
            return data.pos.mean(dim=0, keepdim=True) # Output: [1, 3]
        
        # data.batch is essential for global_mean_pool
        # Ensure data.num_graphs is correctly reflecting the number of graphs for pooling
        num_graphs_for_pool = data.num_graphs if hasattr(data, 'num_graphs') and data.num_graphs > 0 else 1
        if num_graphs_for_pool == 1 and data.batch.max().item() == 0 and data.pos.shape[0] == data.batch.shape[0]:
             # If num_graphs is 1, but batch vector exists and is all zeros, it's a single graph in a batch.
             pass # global_mean_pool handles this.
        elif num_graphs_for_pool == 0 : # Should not happen with valid data
             logger.warning("CentroidBaseline: num_graphs_for_pool is 0.")
             return torch.empty((0,3), dtype=data.pos.dtype, device=data.pos.device)


        return global_mean_pool(data.pos, data.batch, size=num_graphs_for_pool)

class ConvexHullCentroidBaseline(BaseGeometricBaseline):
    def forward(self, data: Batch) -> torch.Tensor:
        pred_coms = []
        
        # Determine the number of graphs to iterate over
        # Handles cases where data might be a single Data object or a Batch object
        if hasattr(data, 'num_graphs') and data.num_graphs > 0:
            num_graphs_to_iterate = data.num_graphs
        elif not hasattr(data, 'batch') or data.batch is None: # Single Data object not in a Batch
            num_graphs_to_iterate = 1
            logger.debug("ConvexHullCentroidBaseline: data.num_graphs not found or 0, assuming single graph.")
        else: # Fallback, though unusual if data is a Batch object
            num_graphs_to_iterate = data.batch.max().item() + 1 if hasattr(data, 'batch') and data.batch is not None else 1
            logger.warning(f"ConvexHullCentroidBaseline: num_graphs inferred as {num_graphs_to_iterate}.")


        if num_graphs_to_iterate == 0:
            logger.warning("ConvexHullCentroidBaseline: num_graphs_to_iterate is 0.")
            return torch.empty((0,3), dtype=data.pos.dtype, device=data.pos.device)

        for i in range(num_graphs_to_iterate):
            if num_graphs_to_iterate == 1 and (not hasattr(data, 'ptr') or data.ptr is None):
                # Single graph, not part of a PyG Batch object from DataLoader, or Batch object with one graph
                points_for_graph_i = data.pos
            elif hasattr(data, 'ptr') and data.ptr is not None: # Standard PyG Batch object
                 points_for_graph_i = data.pos[data.ptr[i]:data.ptr[i+1]]
            else: 
                logger.error("ConvexHullCentroidBaseline: Unexpected data structure in forward pass. Cannot extract points for graph.")
                # Fallback to using all points, which is likely wrong for a batch
                points_for_graph_i = data.pos 

            points_np = points_for_graph_i.cpu().numpy()

            if points_np.shape[0] == 0: # No points for this graph
                com = np.array([0.0, 0.0, 0.0]) # Default COM
                logger.warning(f"ConvexHullCentroidBaseline: Graph {i} has no points.")
            elif points_np.shape[0] < 4: 
                com = points_np.mean(axis=0)
            else:
                try:
                    hull = ConvexHull(points_np)
                    com = hull.points[hull.vertices].mean(axis=0)
                except Exception as e: 
                    logger.warning(f"ConvexHullCentroidBaseline: QhullError for graph {i} (points: {points_np.shape[0]}), falling back to mean. Error: {e}")
                    com = points_np.mean(axis=0)
            pred_coms.append(torch.tensor(com, dtype=data.pos.dtype, device=data.pos.device))
        
        if not pred_coms: 
            logger.error("ConvexHullCentroidBaseline: No COM calculated, returning empty tensor.")
            return torch.empty((0,3), dtype=data.pos.dtype, device=data.pos.device)

        return torch.stack(pred_coms)
