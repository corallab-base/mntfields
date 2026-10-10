import os
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

# Denser contours for earlier structure visibility during online training.
ORIGINAL_CONTOUR_LEVELS = np.arange(0.0, 3.0, 0.03)


class MappingPlotter:
    """
    Lightweight plotting and map export helper for GridmapNode.
    Keeps visualization concerns out of the navigation/training control logic.
    """

    def __init__(self, node):
        self.node = node
        self._region_field_cache = {}

    def save_composite_plot(self):
        n = self.node
        os.makedirs(n.folder, exist_ok=True)
        out_dir = os.path.join(n.folder, "plots2")
        os.makedirs(out_dir, exist_ok=True)

        gm = n.gridmap
        b_map = [
            [gm.origin[0] - gm.map_size / 2, gm.origin[1] - gm.map_size / 2],
            [gm.origin[0] + gm.map_size / 2, gm.origin[1] + gm.map_size / 2],
        ]

        data = gm.data
        vis = gm.visited
        n_cells = gm.n_cells

        bg = np.zeros((n_cells, n_cells, 3), dtype=np.uint8)
        bg[:, :] = [10, 10, 80]      # unknown
        bg[vis == 1] = [220, 220, 40]
        bg[data == 1] = [0, 0, 0]

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        extent = [b_map[0][0], b_map[1][0], b_map[0][1], b_map[1][1]]
        ax.imshow(bg, origin="lower", extent=extent, zorder=0, aspect="auto")

        # Field contours (cache active region every frame; render all cached regions).
        def _compute_region_field(reg, src_world):
            b = reg.bound.detach().cpu().numpy()
            grid_n = 70
            xs = np.linspace(float(b[0, 0]), float(b[1, 0]), grid_n)
            ys = np.linspace(float(b[0, 1]), float(b[1, 1]), grid_n)
            X_w, Y_w = np.meshgrid(xs, ys)

            x0 = np.repeat(src_world[None, :], X_w.size, axis=0)
            x1 = np.stack([X_w.reshape(-1), Y_w.reshape(-1), np.zeros(X_w.size)], axis=1).astype(np.float32)
            xp_world = np.concatenate([x0, x1], axis=1)

            xp_t = torch.tensor(xp_world, dtype=torch.float32)
            xp_n = reg.normalize_tensor(xp_t).to(reg.device)
            with torch.no_grad():
                tt = reg.function.TravelTimes(xp_n)
                if tt.ndim == 2 and tt.shape[1] == 1:
                    tt = tt[:, 0]
                TT = tt.detach().cpu().numpy().reshape(X_w.shape)
            return X_w, Y_w, TT

        # Update active region cache.
        region = n.region
        if region is not None and getattr(region, "function", None) is not None:
            try:
                if n.robot_pos is not None:
                    src = np.array([float(n.robot_pos[0]), float(n.robot_pos[1]), 0.0], dtype=np.float32)
                else:
                    b = region.bound.detach().cpu().numpy()
                    center = 0.5 * (b[0] + b[1])
                    src = np.array([float(center[0]), float(center[1]), 0.0], dtype=np.float32)
                X_w, Y_w, TT = _compute_region_field(region, src)
                self._region_field_cache[region.version] = (X_w, Y_w, TT)
            except Exception:
                pass

        # Seed cache for loaded regions missing cache.
        for reg in n.all_regions.values():
            if reg.version in self._region_field_cache:
                continue
            if getattr(reg, "function", None) is None:
                continue
            try:
                if n.robot_pos is not None:
                    src = np.array(
                        [float(n.robot_pos[0]), float(n.robot_pos[1]), 0.0],
                        dtype=np.float32,
                    )
                else:
                    b = reg.bound.detach().cpu().numpy()
                    center = 0.5 * (b[0] + b[1])
                    src = np.array(
                        [float(center[0]), float(center[1]), 0.0],
                        dtype=np.float32,
                    )
                X_w, Y_w, TT = _compute_region_field(reg, src)
                self._region_field_cache[reg.version] = (X_w, Y_w, TT)
            except Exception:
                pass

        # Draw all cached contours (active + previous regions).
        for ver, cache in list(self._region_field_cache.items()):
            X_w, Y_w, TT = cache
            finite = np.isfinite(TT)
            if not finite.any():
                continue
            tmin = float(np.nanmin(TT[finite]))
            tmax = float(np.nanmax(TT[finite]))
            if tmax - tmin <= 1e-4:
                continue
            levels = ORIGINAL_CONTOUR_LEVELS[
                (ORIGINAL_CONTOUR_LEVELS >= tmin) & (ORIGINAL_CONTOUR_LEVELS <= tmax)
            ]
            if len(levels) < 2:
                continue
            is_active = (region is not None and ver == region.version)
            ax.contour(
                X_w, Y_w, TT,
                levels=levels,
                cmap="magma",
                linewidths=(1.0 if is_active else 0.7),
                alpha=(0.95 if is_active else 0.65),
                zorder=3,
            )

        # Draw active path
        if n.path is not None and len(n.path) >= 2:
            p = np.asarray(n.path, dtype=np.float32)
            ax.plot(p[:, 0], p[:, 1], color="magenta", linewidth=2.0, zorder=9)

        # Draw trajectory
        if len(n._traj_history) > 1:
            tr = np.asarray(n._traj_history, dtype=np.float32)
            ax.plot(tr[:, 0], tr[:, 1], color="cyan", linewidth=1.0, zorder=8)

        # Region bounds
        for reg in n.all_regions.values():
            b = reg.bound.detach().cpu().numpy()
            rect_w = float(b[1, 0] - b[0, 0])
            rect_h = float(b[1, 1] - b[0, 1])
            is_active = (n.region is not None and reg.version == n.region.version)
            edge_color = "yellow" if is_active else "gray"
            lw = 2.0 if is_active else 1.0
            rect = plt.Rectangle(
                (float(b[0, 0]), float(b[0, 1])), rect_w, rect_h,
                linewidth=lw, edgecolor=edge_color, facecolor="none", zorder=5
            )
            ax.add_patch(rect)
            ax.text(float(b[0, 0]) + 0.05, float(b[1, 1]) - 0.12,
                    f"R{reg.version}", color=edge_color, fontsize=8, zorder=6)

        if n.robot_pos is not None:
            ax.scatter(n.robot_pos[0], n.robot_pos[1], c="red", s=100, zorder=10)
        if n.goal_xy is not None:
            ax.scatter(n.goal_xy[0], n.goal_xy[1], c="lime", s=150, marker="*", zorder=10)

        # Viewport:
        # Show the explored workspace (all region bounds + robot/goal/path) so
        # earlier regions never get clipped when the active region moves.
        # Fall back to full map if no region extents are available.
        margin_m = float(n.get_parameter("plot_active_region_margin_m").value)
        pad = max(0.3, margin_m)
        xs = []
        ys = []
        for reg in n.all_regions.values():
            b = reg.bound.detach().cpu().numpy()
            xs.extend([float(b[0, 0]), float(b[1, 0])])
            ys.extend([float(b[0, 1]), float(b[1, 1])])
        if n.robot_pos is not None:
            xs.append(float(n.robot_pos[0]))
            ys.append(float(n.robot_pos[1]))
        if n.goal_xy is not None:
            xs.append(float(n.goal_xy[0]))
            ys.append(float(n.goal_xy[1]))
        if n.path is not None and len(n.path) > 0:
            p = np.asarray(n.path, dtype=np.float32)
            xs.extend(p[:, 0].astype(float).tolist())
            ys.extend(p[:, 1].astype(float).tolist())

        if len(xs) >= 2 and len(ys) >= 2:
            x0 = min(xs) - pad
            x1 = max(xs) + pad
            y0 = min(ys) - pad
            y1 = max(ys) + pad
            # Keep equal scaling by expanding to a square view around center.
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            half = 0.5 * max((x1 - x0), (y1 - y0))
            x0 = cx - half
            x1 = cx + half
            y0 = cy - half
            y1 = cy + half
            # Clamp to map bounds.
            x0 = max(b_map[0][0], x0)
            x1 = min(b_map[1][0], x1)
            y0 = max(b_map[0][1], y0)
            y1 = min(b_map[1][1], y1)
            if x1 > x0 and y1 > y0:
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
            else:
                ax.set_xlim(b_map[0][0], b_map[1][0])
                ax.set_ylim(b_map[0][1], b_map[1][1])
        else:
            ax.set_xlim(b_map[0][0], b_map[1][0])
            ax.set_ylim(b_map[0][1], b_map[1][1])
        ax.set_aspect("equal")
        ax.set_title(f"Step {n._composite_step} | mode={n.mode}/{n.nav_state} | regions={len(n.all_regions)}")

        fig.savefig(os.path.join(out_dir, f"plot_step_{n._composite_step}.png"), dpi=120, bbox_inches="tight")
        fig.savefig(os.path.join(n.folder, "plot2.png"), bbox_inches="tight")
        plt.close(fig)

    def save_ros_map(self, filepath_base: str = None, save_satellites: bool = False):
        import yaml as _yaml
        n = self.node
        gm = n.gridmap
        if filepath_base is None:
            filepath_base = os.path.join(n.folder, "ros_map", "map")
        os.makedirs(os.path.dirname(filepath_base), exist_ok=True)

        data = gm.data.copy()
        vis = gm.visited.copy()
        n_cells = gm.n_cells

        pgm = np.full((n_cells, n_cells), 205, dtype=np.uint8)
        pgm[(data == 0) & (vis == 1)] = 254
        pgm[data == 1] = 0
        pgm = np.flipud(pgm)

        pgm_path = filepath_base + ".pgm"
        yaml_path = filepath_base + ".yaml"
        with open(pgm_path, "wb") as f:
            f.write(f"P5\n{n_cells} {n_cells}\n255\n".encode())
            f.write(pgm.tobytes())

        origin_x = float(gm.origin[0] - gm.map_size / 2.0)
        origin_y = float(gm.origin[1] - gm.map_size / 2.0)
        meta = {
            "image": os.path.basename(pgm_path),
            "resolution": float(gm.cell_size),
            "origin": [origin_x, origin_y, 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }
        with open(yaml_path, "w") as f:
            _yaml.dump(meta, f, default_flow_style=False)

        n.get_logger().info(f"[MAP SAVED] {pgm_path}")

        if save_satellites:
            for ver, reg in n.all_regions.items():
                if n.region is not None and ver == n.region.version:
                    continue
                try:
                    if getattr(reg, "loaded", True):
                        reg.save_model()
                except Exception as e:
                    n.get_logger().warn(f"[MAP SAVED] Could not save satellite R{ver}: {e}")

        n._save_topology()

    def save_occupancy_snapshot(self, tag=None):
        n = self.node
        out_dir = os.path.join(n.folder, "occupancy_snapshots")
        os.makedirs(out_dir, exist_ok=True)
        if tag is None:
            tag = f"step_{n._composite_step:06d}"

        data = n.gridmap.data.copy()
        visited = n.gridmap.visited.copy()
        frontier = n.gridmap.frontier.copy()
        n_cells = n.gridmap.n_cells

        img = np.zeros((n_cells, n_cells, 3), dtype=np.uint8)
        img[:, :] = [10, 10, 80]
        img[visited == 1] = [200, 200, 0]
        img[data == 1] = [0, 0, 0]
        img[frontier == 1] = [255, 140, 0]

        plt.figure(figsize=(8, 8))
        plt.imshow(img, origin="lower")
        plt.title(f"Occupancy Snapshot {tag}")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        np.savez_compressed(
            os.path.join(out_dir, f"{tag}.npz"),
            occupancy=data,
            visited=visited,
            frontier=frontier,
        )
