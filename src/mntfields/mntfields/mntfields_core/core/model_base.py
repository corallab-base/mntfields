import os
import copy
import math
import torch
import numpy as np

from mntfields.mntfields_core.ntrl import model_network_metric as model_network
from mntfields.mntfields_core.ntrl import model_function_metric as model_function

class Model:
    def __init__(
        self,
        folder: str,
        dim: int,
        B_scale: float,
        device: str = "cuda:0",
        dim_cells: int = 128,
        init_network: bool = True,
        eval: bool = False,
        lr: float = 1e-4,
    ):
        """
        Base model for neural metric learning.
        
        Args:
            folder: str - save directory
            dim: int - spatial dimension (typically 3 for x,y,z)
            B_scale: float - Fourier feature scale
            device: str - cuda device
            dim_cells: int - occupancy grid resolution
            init_network: bool - whether to initialize network
            eval: bool - evaluation mode
            lr: float - learning rate
        """
        self.folder = folder
        self.dim = dim
        self.B_scale = B_scale
        
        # Device setup — fall back to CPU if CUDA is unavailable or broken
        if device.startswith("cuda") and not torch.cuda.is_available():
            import warnings
            warnings.warn(
                f"[Model] Requested device '{device}' but CUDA is not available "
                "(torch.cuda.is_available() == False). Falling back to CPU.",
                RuntimeWarning,
            )
            device = "cpu"
        self.Params = {}
        self.Params['Device'] = device
        self.Params['Pytorch Amp'] = False
        self.device = device
        # Network parameters
        self.Params['Network'] = {}
        self.Params['Network']['Normalisation'] = "OffsetMinMax"
        
        # Training parameters
        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points'] = 2e5
        self.Params['Training']['Batch Size'] = 2000
        self.Params['Training']['Validation Percentage'] = 10
        self.Params['Training']['Number of Epochs'] = 20000
        self.Params['Training']['Resampling Bounds'] = (0.1, 0.9)
        self.Params['Training']['Print Every Epoch'] = 1
        self.Params['Training']['Save Every Epoch'] = 50
        self.Params['Training']['Learning Rate'] = lr
        self.Params['Training']['Random Distance Sampling'] = True
        self.Params['Training']['Use Scheduler'] = False
        
        # Training state
        self.total_train_loss = []
        self.total_val_loss = []
        self.epoch = 0
        self.frame_idx = 0
        
        # Network components (will be initialized if init_network=True)
        self.network = None
        self.function = None
        self.optimizer = None
        self.scheduler = None
        self.B = None
        
        # Training utilities
        self.prev_state_queue = []
        self.prev_loss_queue = []
        self.prev_optimizer_queue = []
        self.timer = []
        
        # Initialize network if requested
        if init_network:
            self.init_network()
    
    def init_network(self):
        """
        Initialize neural network, optimizer, and function wrapper.
        This method initializes:
        1. Fourier feature matrix B
        2. Neural network (metric field)
        3. Function wrapper (loss, speed, gradient queries)
        4. Optimizer
        """
        print("Initializing network...")
        
        # Import here to avoid circular dependency
        #from mntfields.mntfields_core.ntrl import model_network_metric as model_network
        #from mntfields.mntfields_core.ntrl import model_function_metric as model_function
        
        # 1. Initialize Fourier feature matrix
        self.B = torch.normal(0, 1, size=(128, self.dim))
        torch.nn.init.trunc_normal_(
            self.B, 
            mean=0.0, 
            std=1.0, 
            a=-2.0, 
            b=2.0
        )
        
        # 2. Initialize network
        self.network = model_network.NN(
            self.Params['Device'],
            self.dim,
            self.B
        )
        self.network.apply(self.network.init_weights)
        self.network.float()
        self.network.to(self.Params['Device'])
        
        # 3. Initialize function wrapper
        self.function = model_function.Function(
            self.folder,
            self.Params['Device'],
            self.network,
            self.dim
        )
        
        # 4. Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.Params['Training']['Learning Rate'],
            weight_decay=0.1
        )
        
        # 5. Initialize scheduler (if enabled)
        if self.Params['Training']['Use Scheduler']:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer
            )
        
        print(f"✅ Network initialized: {self.network}")
        print(f"   - B shape: {self.B.shape}")
        print(f"   - Device: {self.Params['Device']}")
        print(f"   - LR: {self.Params['Training']['Learning Rate']}")

    # --------------------------------------------------
    # Delete network (for region unload)
    # --------------------------------------------------

    def delete_network(self):
        self.network = None
        self.function = None
        self.optimizer = None

    # --------------------------------------------------
    # Core training
    # --------------------------------------------------
#    def train_core(self, epoch, frame_data=None, is_one_frame=True):
#        """
#        Core training loop with batching, speed transformation, and state rollback.
#        Matches original mNTFields training protocol.
#        """
#        if frame_data is None or len(frame_data) == 0:
#            return None, None
#    
#        beta = 1.0
#        prev_diff = 1.0
#        current_diff = 1.0
#    
#        # Move data to device
#        cur_data = frame_data.to(self.device)
#    
#        # Create dataloader with batching
#        batch_size = 5000  # Adjust based on GPU memory
#        dataloader = FastTensorDataLoader(cur_data, batch_size=batch_size, shuffle=True)
#    
#        # Multiple training epochs on same data
#        frame_epoch = epoch
#        all_loss = []
#    
#        for e in range(frame_epoch):
#            total_train_loss = 0.0
#            total_diff = 0.0
#    
#            # Save current state for potential rollback
#            if not hasattr(self, 'prev_state_queue'):
#                self.prev_state_queue = []
#                self.prev_optimizer_queue = []
#    
#            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
#            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
#            self.prev_state_queue.append(current_state)
#            self.prev_optimizer_queue.append(current_optimizer)
#    
#            if len(self.prev_state_queue) > 5:
#                self.prev_state_queue.pop(0)
#                self.prev_optimizer_queue.pop(0)
#    
#            # Set learning rate
#            self.optimizer.param_groups[0]['lr'] = 5e-4
#    
#            prev_diff = current_diff
#            iter_count = 0
#    
#            # Inner loop with divergence detection
#            while True:
#                total_train_loss = 0.0
#                total_diff = 0.0
#                valid_batches = 0
#    
#                # Process batches
#                for i, data in enumerate(dataloader):
#                    if i > 15:  # Limit batches per epoch
#                        break
#    
#                    data = data[0].to(self.device)
#    
#                    # Split data
#                    points = data[:, :2 * self.dim].float()              # [x0, x1]
#                    speed = data[:, 2 * self.dim:2 * self.dim + 2].float()  # [y0, y1]
#                    normal = data[:, 2 * self.dim + 2:].float()          # [n0, n1]
#    
#                    # Apply speed transformation (critical!)
#                    speed = speed * speed * (2 - speed) * (2 - speed)
#    
#                    # Alpha blending (if alpha exists)
#                    if hasattr(self, 'alpha'):
#                        speed = self.alpha * speed + 1 - self.alpha
#    
#                    # Input NaN guard
#                    if torch.isnan(points).any() or torch.isnan(speed).any() or torch.isnan(normal).any():
#                        print("[MODELBASE] NaN in batch inputs, skipping batch")
#                        continue
#    
#                    gamma = 0.001
#                    out = self.function.Loss(points, speed, normal, beta, gamma, epoch)
#    
#                    if out is None:
#                        # Loss chose to skip this batch
#                        continue
#    
#                    if isinstance(out, tuple):
#                        loss_value, loss_n, wv = out
#                    else:
#                        loss_value = out
#                        loss_n = out
#                        wv = None
#    
#                    # Loss may explicitly signal skip
#                    if loss_value is None:
#                        continue
#    
#                    # Ensure tensor
#                    if not torch.is_tensor(loss_value):
#                        loss_value = torch.tensor(
#                            float(loss_value), device=self.device, dtype=torch.float32
#                        )
#    
#                    if loss_n is None:
#                        loss_n = loss_value
#    
#                    # This batch is valid
#                    valid_batches += 1
#    
#                    # Backward pass
#                    loss_value.backward()
#                    self.optimizer.step()
#                    self.optimizer.zero_grad()
#    
#                    # Accumulate loss
#                    total_train_loss += float(loss_value.detach().item())
#                    total_diff += float(loss_n.detach().item())
#    
#                    del points, speed, normal, loss_value, loss_n, wv
#    
#                # If no valid batch, skip this frame instead of reinitializing
#                if valid_batches == 0:
#                    print("⚠️  All batches invalid (Loss returned None). Skipping frame.")
#                    # Do not treat as fatal; just keep last_loss unchanged
#                    return 0.0, cur_data
#    
#                # Average over batches
#                total_train_loss /= valid_batches
#                total_diff /= valid_batches
#    
#                # Check for NaN in averaged diff
#                if not np.isfinite(total_diff):
#                    print("⚠️  NaN/Inf detected in total_diff, skipping frame.")
#                    return None, None
#    
#                current_diff = total_diff
#                diff_ratio = current_diff / (prev_diff + 1e-12)
#    
#                # Check convergence / rollback
#                if (0 < diff_ratio < 1.2) or e < 10:
#                    break
#                else:
#                    iter_count += 1
#                    if iter_count > 20:
#                        break
#    
#                    with torch.no_grad():
#                        idx = random.randint(0, len(self.prev_state_queue) - 1)
#                        self.network.load_state_dict(self.prev_state_queue[idx], strict=True)
#                        self.optimizer.load_state_dict(self.prev_optimizer_queue[idx])
#    
#                    print(f"Rollback: Epoch {e}, Loss = {total_diff:.4e}")
#    
#            all_loss.append(total_diff)
#    
#            # Early stopping
#            if total_diff < 0.001:
#                print(f"✅ Converged at epoch {e}, loss = {total_diff:.6f}")
#                break
#    
#            # Update beta
#            beta = 1.0 / (total_diff + 1e-12)
#    
#            # Scheduler step
#            if self.scheduler is not None:
#                self.scheduler.step(total_train_loss)
#    
#            self.last_loss = float(total_diff)
#    
#        return total_diff, cur_data

    def train_core(self, epoch, frame_data=None, is_one_frame=True):

        if frame_data is None or len(frame_data) == 0:
            return None, None

        # Guard against race condition where _unload_current_region() sets
        # optimizer=None while this method is running in a background thread.
        if self.optimizer is None or self.network is None:
            return None, None

        beta       = 1.0
        prev_diff  = 1.0
        current_diff = 1.0
        # Initialise so return is defined even when epoch=0
        total_train_loss = 0.0
        total_diff       = 0.0

        cur_data   = frame_data.to(self.device)
        n          = cur_data.shape[0]
        batch_size = 5000
        max_batches = 6

        for e in range(epoch):

            total_train_loss = 0.0
            total_diff       = 0.0

            if not hasattr(self, 'prev_state_queue'):
                self.prev_state_queue = []
                self.prev_optimizer_queue = []
                self.prev_loss_queue = []

            # ── Clone weights on-device — avoids pickle CPU serialisation ──
            current_state     = {k: v.detach().clone()
                                 for k, v in self.network.state_dict().items()}
            current_optimizer = copy.deepcopy(self.optimizer.state_dict())
            self.prev_state_queue.append(current_state)
            self.prev_optimizer_queue.append(current_optimizer)
            self.prev_loss_queue.append(current_diff)

            if len(self.prev_state_queue) > 5:
                self.prev_state_queue.pop(0)
                self.prev_optimizer_queue.pop(0)
                self.prev_loss_queue.pop(0)

            self.optimizer.param_groups[0]['lr'] = self.Params['Training']['Learning Rate']

            prev_diff  = current_diff
            iter_count = 0

            while True:

                total_train_loss = 0.0
                total_diff       = 0.0

                # ── GPU shuffle — no Python DataLoader overhead ──────────
                perm    = torch.randperm(n, device=self.device)
                shuffled = cur_data[perm]          # one permuted copy on GPU
                n_batches = min(max_batches, math.ceil(n / batch_size))

                for i in range(n_batches):
                    data = shuffled[i * batch_size : (i + 1) * batch_size]

                    points = data[:, :2*self.dim].float()
                    speed  = data[:, 2*self.dim : 2*self.dim + 2].float()
                    normal = data[:, 2*self.dim + 2:].float()

                    speed = speed * speed * (2 - speed) * (2 - speed)

                    if hasattr(self, 'alpha'):
                        speed = self.alpha * speed + 1 - self.alpha

                    gamma = 0.001
                    out   = self.function.Loss(points, speed, normal, beta, gamma, epoch)

                    if isinstance(out, tuple):
                        loss_value, loss_n, wv = out
                    else:
                        loss_value = out
                        loss_n     = out
                        wv         = None

                    loss_value.backward()
                    if self.optimizer is None:   # unloaded mid-training
                        return None, None
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # ── Accumulate as Python scalars — avoids GPU tensor
                    #    accumulation and implicit CPU-GPU sync per step ──
                    total_train_loss += loss_value.detach().item()
                    total_diff       += loss_n.detach().item()

                total_train_loss /= n_batches
                total_diff       /= n_batches

                # ── Pure-Python NaN check — no GPU-CPU sync needed ───────
                if not math.isfinite(total_diff):
                    print("⚠️ NaN detected")
                    return None, None

                current_diff = total_diff
                diff_ratio   = current_diff / (prev_diff + 1e-12)

                if (diff_ratio < 1.2 and diff_ratio > 0) or e < 10:
                    break
                else:
                    iter_count += 1
                    if iter_count > 20:
                        break

                    with torch.no_grad():
                        # Restore best state (lowest loss) instead of random
                        best_idx = min(range(len(self.prev_loss_queue)),
                                       key=lambda i: self.prev_loss_queue[i])
                        self.network.load_state_dict(
                            self.prev_state_queue[best_idx], strict=True)
                        self.optimizer.load_state_dict(
                            self.prev_optimizer_queue[best_idx])

            if total_diff < 0.001:
                break

            beta = 1.0 / (total_diff + 1e-12)

            if self.scheduler is not None:
                self.scheduler.step(total_train_loss)

        self.last_loss = total_diff
        return total_diff, None
