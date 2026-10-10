import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

class Function:
    def __init__(self, path: str, device: str, network, dim: int):
        self.path = path
        self.device = device
        self.network = network
        self.dim = dim
        
        self.total_train_loss = []
        self.total_val_loss = []
        
        # Hyperparameters
        self.alpha = 1.025
        limit = 0.5
        self.margin = limit / 15.0
        self.offset = self.margin / 10.0
    
    def gradient(self, y: Tensor, x: Tensor, create_graph=True) -> Tensor:
        """
        Computes ∂y/∂x for batched inputs.

        Used everywhere because:
        The travel-time field τ must behave like a differentiable distance field.
        """
        grad_y = torch.ones_like(y)
        grad_x = torch.autograd.grad(
            y, x, grad_y,
            only_inputs=True,
            retain_graph=True,
            create_graph=create_graph
        )[0]
        return grad_x

    # ============================================================
    # QUERY FUNCTIONS (for visualization and planning)
    # ============================================================
    
    def TravelTimes(self, Xp):
        """
        Returns predicted travel time τ(x0, x1).

        This is the scalar field your robot follows.
        The planner uses ∇τ to decide direction.
        """
        tau, w, coords = self.network.out(Xp)
        TT = tau[:, 0]
        return TT

    def Speed(self, Xp):
        """
        Speed is defined as:

            speed = 1 / ||∇τ||

        Because in optimal control:
            |∇τ| = 1 / speed

        So this converts gradient magnitude into speed prediction.
        """
        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        DT0 = dtau[:, :self.dim]
        # ||∇τ||²
        S = torch.einsum('ij,ij->i', DT0, DT0)

        # speed = 1 / ||∇τ||
        Ypred = 1 / torch.sqrt(S)

        del Xp, tau, dtau
        return Ypred
        
    def Gradient(self, Xp):
        """
        Returns normalized direction for planning.

        The robot follows:

            direction = -∇τ

        because travel-time decreases along that direction.
        """
        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        
        # start point gradient
        Ypred0 = -dtau[:, :self.dim]
        Spred0 = torch.norm(Ypred0, dim=1).view(-1, 1)
        Ypred0 = 1 / (Spred0 ** 2) * Ypred0
        
        # end point gradient
        Ypred1 = -dtau[:, self.dim:]
        Spred1 = torch.norm(Ypred1, dim=1).view(-1, 1)
        Ypred1 = 1 / (Spred1 ** 2) * Ypred1
        
        return torch.cat((Ypred0, Ypred1), dim=1)
    
    # ============================================================
    # LOSS FUNCTION (for training)
    # ============================================================
    
    def Loss(self, points, Yobs, normal, beta, gamma, epoch):
        """
        Full mNTFields loss.
        """
        n = Yobs.shape[0]   # Python int — avoids repeated shape attribute access
        # Forward pass
        tau, w, Xp = self.network.out(points)
        
        # Clamp tau
        #tau = torch.clamp(tau, min=1e-4, max=10.0)
        
        dtau = self.gradient(tau, Xp)

        # Gradient at start and end
        DT0 = dtau[:, :self.dim]
        DT1 = dtau[:, self.dim:]
        
        # Squared gradient norms
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)
        
        # Temporal difference loss
        td_weight = 1e-3
        
        with torch.no_grad():
            #length0 = torch.clamp(0.03 / (Yobs[:, 0].unsqueeze(1) + 1e-4), min=1e-4, max=0.5)
            length0 = (0.03)/(Yobs[:,0]).unsqueeze(1)
            Dir0 = length0 * (DT0 * Yobs[:, 0].unsqueeze(1) ** 2)
            
            Xpnew0 = Xp.clone().detach()
            Xpnew0[:, :self.dim] = Xpnew0[:, :self.dim] - Dir0
            
            taunew0, w, Xpnew0 = self.network.out(Xpnew0)
            #taunew0 = torch.clamp(taunew0, min=1e-4, max=10.0)
            taunew1 = length0
            del Xpnew0, Dir0
        tau_loss0 = td_weight * ((tau - taunew0 - taunew1) ** 2).squeeze()
        
        with torch.no_grad():
            #length1 = torch.clamp(0.03 / (Yobs[:, 1].unsqueeze(1) + 1e-4), min=1e-4, max=0.5)
            length1 = (0.03)/(Yobs[:,1]).unsqueeze(1)
            Dir1 = length1 * (DT1 * Yobs[:, 1].unsqueeze(1) ** 2)
            
            Xpnew0 = Xp.clone().detach()
            Xpnew0[:, self.dim:] = Xpnew0[:, self.dim:] - Dir1
            
            taunew0, w, Xpnew0 = self.network.out(Xpnew0)
            #taunew0 = torch.clamp(taunew0, min=1e-4, max=10.0)
            taunew1 = length1
            del Xpnew0, Dir1
        tau_loss1 = td_weight * ((tau - taunew0 - taunew1) ** 2).squeeze()
        
        # Ignore very small distances
        where_d0 = (tau[:, 0] > length0.squeeze())
        where_d1 = (tau[:, 0] > length1.squeeze())
        tau_loss0[~where_d0] = 0
        tau_loss1[~where_d1] = 0
        tau_loss = tau_loss0 + tau_loss1
        
        # Speed prediction loss
        Ypred0 = torch.sqrt(S0 + 1e-8)#(torch.clamp(S0, min=1e-8, max=1e4))
        Ypred1 = torch.sqrt(S1 + 1e-8)#torch.clamp(S1, min=1e-8, max=1e4))
        
        sq_Ypred0 = Ypred0
        sq_Ypred1 = Ypred1
        #sq_Yobs0 = torch.clamp(Yobs[:, 0], min=1e-4, max=10.0)
        #sq_Yobs1 = torch.clamp(Yobs[:, 1], min=1e-4, max=10.0)
        sq_Yobs0 = (Yobs[:,0])
        sq_Yobs1 = (Yobs[:,1])
        
        l0 = sq_Yobs0 * sq_Ypred0
        l1 = sq_Yobs1 * sq_Ypred1
        
        l02 = torch.sqrt(l0)#torch.clamp(l0, min=0.0))
        l12 = torch.sqrt(l1)#torch.clamp(l1, min=0.0))
        
        loss_weight = 1e-2
        loss0 = loss_weight * (l02-1)**2#torch.clamp((l02 - 1) ** 2, max=100.0)
        loss1 = loss_weight * (l12-1)**2#torch.clamp((l12 - 1) ** 2, max=100.0)
        T = tau[:,0]
        diff = loss0 + loss1
        
        # Normal alignment loss
        normal_weight = 1e-3
        normal0 = normal[:, :self.dim]
        normal1 = normal[:, self.dim:]
        
        n_loss0 = (1.001 - Yobs[:, 0].unsqueeze(1)) * (
            Yobs[:, 0].unsqueeze(1) * DT0 + normal0
        ) ** 2
        n_loss1 = (1.001 - Yobs[:, 1].unsqueeze(1)) * (
            Yobs[:, 1].unsqueeze(1) * DT1 + normal1
        ) ** 2
        n_loss = normal_weight * (
            torch.sum(n_loss0, dim=1) + torch.sum(n_loss1, dim=1)
        )
        #n_loss = torch.clamp(n_loss, max=100.0)
        para = 0.5
        #lossn = (torch.sum((diff+n_loss + tau_loss)*torch.exp(-0.5*T)))/Yobs.shape[0] uncomment
        exp_arg = torch.clamp(-0.15 * T, min=-20.0, max=20.0)  # 0.5→0.15: gives more weight to long-range pairs so the field learns topology across the room
        w = torch.exp(exp_arg)
        lossn = torch.sum((diff + n_loss + tau_loss) * w) / n
        #print("[LOSS DEBUG] entering Loss, epoch", epoch,
        #      "points", points.shape, "Yobs", Yobs.shape)
#
        #print("[LOSS DEBUG] diff[min,max] =",
        #      float(diff.min().item()), float(diff.max().item()))
        #print("[LOSS DEBUG] n_loss[min,max] =",
        #      float(n_loss.min().item()), float(n_loss.max().item()))
        #print("[LOSS DEBUG] tau_loss[min,max] =",
        #      float(tau_loss.min().item()), float(tau_loss.max().item()))
        #print("[LOSS DEBUG] T[min,max] =",
        #      float(T.min().item()), float(T.max().item()))
        #print("[LOSS DEBUG] w[min,max] =",
        #      float(w.min().item()), float(w.max().item()))
        #print("[LOSS DEBUG] lossn type =", type(lossn),
        #      "is_tensor =", torch.is_tensor(lossn))
        ## Total loss
        ##T = torch.clamp(tau[:, 0], min=1e-4)
        ##lossn = torch.sum(
        ##    (diff + n_loss + tau_loss) * torch.exp(-0.5 * T)
        ##) / Yobs.shape[0]
        #if torch.is_tensor(lossn):
        #    bad = torch.isnan(lossn).any() or torch.isinf(lossn).any()
        #if bad:
        #    print("LOSS INVALID, tau[min,max] =",
        #          float(tau.min().item()), float(tau.max().item()),
        #          "T[min,max] =", float(T.min().item()), float(T.max().item()))
        #    print("LOSS INVALID, S0[min,max] =",
        #          float(S0.min().item()), float(S0.max().item()),
        #          "S1[min,max] =", float(S1.min().item()), float(S1.max().item()))
        #    print("LOSS INVALID, Yobs[min,max] =",
        #          float(Yobs.min().item()), float(Yobs.max().item()))
        #    return None, None, None
        #else:
        #    print("LOSS DEBUG: lossn is not a tensor, value =", lossn)
        #    # force conversion so backward works:
        #    lossn = torch.tensor(float(lossn), device=points.device, dtype=torch.float32)
#
        ## Check for NaN
        #if torch.isnan(lossn).any() or torch.isinf(lossn).any():
        #    print("LOSS INVALID, tau[min,max] =",
        #          float(tau.min().item()), float(tau.max().item()),
        #          "T[min,max] =", float(T.min().item()), float(T.max().item()))
        #    print("LOSS INVALID, S0[min,max] =", float(S0.min().item()), float(S0.max().item()),
        #          "S1[min,max] =", float(S1.min().item()), float(S1.max().item()))
        #    print("LOSS INVALID, Yobs[min,max] =",
        #          float(Yobs.min().item()), float(Yobs.max().item()))
        #    return None, None, None
#
        #if torch.isnan(lossn).any() or torch.isinf(lossn).any():
        #    print("[LOSS INVALID]",
        #          "tau_min/max", float(tau.min().item()), float(tau.max().item()),
        #          "T_min/max", float(T.min().item()), float(T.max().item()))
        #    return None, None, None
        
        loss = beta * lossn
        return lossn, lossn, diff 
        #return loss, lossn, diff
#    def Loss(self, points, Yobs, normal, beta, gamma, epoch):
#        """
#        Full mNTFields loss.
#        """
#        # Forward pass
#        tau, w_logits, Xp = self.network.out(points)  # keep 'w' name free
#        # tau: [N,1], Xp: [N, 2*dim]
#    
#        dtau = self.gradient(tau, Xp)
#    
#        # Distance between start and end (not directly used in loss here)
#        D = torch.norm(Xp[:, self.dim:] - Xp[:, :self.dim], p=2, dim=1)
#    
#        # Gradient at start and end
#        DT0 = dtau[:, :self.dim]
#        DT1 = dtau[:, self.dim:]
#    
#        # Squared gradient norms
#        S0 = torch.einsum("ij,ij->i", DT0, DT0)
#        S1 = torch.einsum("ij,ij->i", DT1, DT1)
#    
#        # ------------------------------------------------------------------
#        # Temporal difference loss
#        # ------------------------------------------------------------------
#        td_weight = 1e-3
#    
#        with torch.no_grad():
#            # start point
#            length0 = 0.03 / (Yobs[:, 0].unsqueeze(1) + 1e-8)
#            Dir0 = length0 * (DT0 * (Yobs[:, 0].unsqueeze(1) ** 2)).detach()
#    
#            Xpnew0 = Xp.detach().clone()
#            Xpnew0[:, :self.dim] = Xpnew0[:, :self.dim] - Dir0
#    
#            taunew0, _, _ = self.network.out(Xpnew0)
#            taunew1 = length0
#            tau_loss0 = td_weight * (tau - taunew0 - taunew1).squeeze() ** 2
#    
#        with torch.no_grad():
#            # end point
#            length1 = 0.03 / (Yobs[:, 1].unsqueeze(1) + 1e-8)
#            Dir1 = length1 * (DT1 * (Yobs[:, 1].unsqueeze(1) ** 2)).detach()
#    
#            Xpnew1 = Xp.detach().clone()
#            Xpnew1[:, self.dim:] = Xpnew1[:, self.dim:] - Dir1
#    
#            taunew0b, _, _ = self.network.out(Xpnew1)
#            taunew1b = length1
#            tau_loss1 = td_weight * (tau - taunew0b - taunew1b).squeeze() ** 2
#    
#        # Ignore very small distances
#        T = tau[:, 0]  # [N]
#        where_d0 = T > length0.squeeze()
#        where_d1 = T > length1.squeeze()
#        tau_loss0 = tau_loss0 * where_d0.float()
#        tau_loss1 = tau_loss1 * where_d1.float()
#        tau_loss = tau_loss0 + tau_loss1
#    
#        # ------------------------------------------------------------------
#        # Speed prediction loss
#        # ------------------------------------------------------------------
#        Ypred0 = torch.sqrt(S0 + 1e-8)
#        Ypred1 = torch.sqrt(S1 + 1e-8)
#    
#        sq_Yobs0 = Yobs[:, 0]
#        sq_Yobs1 = Yobs[:, 1]
#    
#        l0 = sq_Yobs0 * Ypred0
#        l1 = sq_Yobs1 * Ypred1
#    
#        l0_2 = torch.sqrt(torch.clamp(l0, min=0.0))
#        l1_2 = torch.sqrt(torch.clamp(l1, min=0.0))
#    
#        loss_weight = 1e-2
#        loss0 = loss_weight * (l0_2 - 1.0) ** 2
#        loss1 = loss_weight * (l1_2 - 1.0) ** 2
#    
#        diff = loss0 + loss1  # [N]
#    
#        # ------------------------------------------------------------------
#        # Normal alignment loss
#        # ------------------------------------------------------------------
#        normal_weight = 1e-3
#        normal0 = normal[:, :self.dim]
#        normal1 = normal[:, self.dim:]
#    
#        n_loss0 = (1.001 - Yobs[:, 0].unsqueeze(1)) * (
#            Yobs[:, 0].unsqueeze(1) * DT0 + normal0
#        ) ** 2
#        n_loss1 = (1.001 - Yobs[:, 1].unsqueeze(1)) * (
#            Yobs[:, 1].unsqueeze(1) * DT1 + normal1
#        ) ** 2
#    
#        n_loss = normal_weight * (
#            torch.sum(n_loss0, dim=1) + torch.sum(n_loss1, dim=1)
#        )  # [N]
#    
#        # ------------------------------------------------------------------
#        # Combine with exponential weighting in T
#        # ------------------------------------------------------------------
#        exp_arg = torch.clamp(-0.5 * T, min=-20.0, max=20.0)
#        w = torch.exp(exp_arg)  # [N]
#    
#        lossn = torch.sum((diff + n_loss + tau_loss) * w) / Yobs.shape[0]  # scalar tensor
#    
#        # Debug (optional – you can keep or remove)
#        print("[LOSS DEBUG] entering Loss, epoch", epoch,
#              "points", points.shape, "Yobs", Yobs.shape)
#        print("[LOSS DEBUG] diff[min,max] =",
#              float(diff.min().item()), float(diff.max().item()))
#        print("[LOSS DEBUG] n_loss[min,max] =",
#              float(n_loss.min().item()), float(n_loss.max().item()))
#        print("[LOSS DEBUG] tau_loss[min,max] =",
#              float(tau_loss.min().item()), float(tau_loss.max().item()))
#        print("[LOSS DEBUG] T[min,max] =",
#              float(T.min().item()), float(T.max().item()))
#        print("[LOSS DEBUG] w[min,max] =",
#              float(w.min().item()), float(w.max().item()))
#        print("[LOSS DEBUG] lossn type =", type(lossn),
#              "is_tensor =", torch.is_tensor(lossn),
#              "requires_grad =", lossn.requires_grad)
#    
#        # NaN / Inf guard
#        if torch.isnan(lossn).any() or torch.isinf(lossn).any():
#            print("LOSS INVALID, tau[min,max] =",
#                  float(tau.min().item()), float(tau.max().item()),
#                  "T[min,max] =", float(T.min().item()), float(T.max().item()))
#            print("LOSS INVALID, S0[min,max] =",
#                  float(S0.min().item()), float(S0.max().item()),
#                  "S1[min,max] =", float(S1.min().item()), float(S1.max().item()))
#            print("LOSS INVALID, Yobs[min,max] =",
#                  float(Yobs.min().item()), float(Yobs.max().item()))
#            return None, None, None
#    
#        loss = beta * lossn  # still a scalar tensor with grad
#    
#        print("[LOSS DEBUG] returning loss, lossn, diff:",
#              float(lossn.detach().item()))
#        # Return loss as first element so Model.train_core uses it for backward()
#        return loss, lossn, diff

