"""
Deployment loop for the `deco_vitac` model (the ManiSkill-ViTac 2026 config family:
config/deco_vitac2026_vis.yaml / config/deco_vitac2026_tactile.yaml).

This is modeled on two existing control loops:
  - This repo's own deploy/deploy_h1.py (predict_action / H1_2 arm+hand loop shape,
    fps-controlled while-loop, receding-horizon action queue, ACTTemporalEnsembler).
  - VB-VLA's deploy_scripts/eval_real_bimanual_vb.py (chunk-based receding-horizon
    inference with precise timing + latency compensation, async ObsSaver for
    debugging without blocking the control loop).

What this script owns: model loading, the observation -> language-conditioned
flow-matching inference -> action-chunk-queue loop, receding-horizon/temporal-
ensemble action selection, loop timing, and optional async observation logging.

What this script does NOT own: how to actually get camera/tactile frames and robot
state (`RobotEnv.get_obs`), or how to turn a predicted 20-D action into low-level
robot commands (`RobotEnv.send_action`). Those are hardware/simulator specific and
this repo has no bindings for whatever rig ManiSkill-ViTac 2026 tasks actually run
on -- implement `RobotEnv` for your actual setup (mirror deploy_h1.py's H1_2 arm/
hand/camera calls if it's the same physical robot, or VB-VLA's BimanualUmiEnv if
you're on similar UMI-style hardware).

Action contract (see config/deco_vitac2026_*.yaml's `action_dim: 20` comment): per
arm, 3-D delta-xyz + 6-D rotation delta (continuous 6D representation, Zhou et al.
2019) + 1-D absolute gripper width, x2 arms. `rotation6d_to_matrix`/`compose_delta_pose`
below implement the generic (hardware-independent) half of turning that into a
target end-effector pose; RobotEnv.send_action is responsible for the rest (IK,
joint commands, or whatever your controller expects).
"""

import os
import sys
import time
import json
import yaml
import argparse
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from inference import predict_action_vitac, modeling, ACTTemporalEnsembler  # noqa: E402

TACTILE_KEYS = ("tactile_left_0", "tactile_right_0", "tactile_left_1", "tactile_right_1")


def rotation6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Zhou et al. 2019 continuous 6D rotation representation -> 3x3 rotation matrix.

    d6: (6,) = concat(a1, a2), each a 3-vector. Gram-Schmidt orthogonalize to get an
    orthonormal basis; this is the standard hardware-independent decode, distinct
    from whatever IK/joint-space step your RobotEnv.send_action does afterward.
    """
    a1, a2 = d6[:3], d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # columns = basis vectors


def compose_delta_pose(current_pos: np.ndarray, current_rot: np.ndarray, delta10: np.ndarray):
    """Apply one arm's 10-D chunk of predicted action to its current pose.

    Args:
        current_pos: (3,) current end-effector position.
        current_rot: (3, 3) current end-effector rotation matrix.
        delta10: (10,) = [delta_xyz(3), rotation6d_delta(6), abs_gripper_width(1)],
            already denormalized (see inference.postprocess).
    Returns:
        target_pos: (3,), target_rot: (3, 3), gripper_width: float
    """
    delta_xyz = delta10[:3]
    delta_rot = rotation6d_to_matrix(delta10[3:9])
    gripper_width = float(delta10[9])
    target_pos = current_pos + delta_xyz
    target_rot = current_rot @ delta_rot
    return target_pos, target_rot, gripper_width


class RobotEnv:
    """Hardware/simulator interface -- fill in for your actual ManiSkill-ViTac 2026 rig.

    Not implemented here on purpose: this repo has camera/robot bindings for the
    H1_2 humanoid (see deploy_h1.py's teleimager/robot_control imports) but ManiSkill-
    ViTac 2026 may run on different arms or purely in simulation. Wire whichever of
    those (or a ManiSkill3 env's .reset()/.step()) matches your setup.
    """

    def get_obs(self) -> dict:
        """Return a dict with keys:
            img1, img2: HWC uint8 RGB arrays (observation.images.camera0/1)
            obs: (20,) float array, current proprioceptive state
            tactile_imgs: dict of the 4 TACTILE_KEYS -> HWC uint8 RGB arrays
                (omit / pass None if use_tactile is False in the config)
            current_pos_left, current_rot_left: (3,), (3,3) -- current left EE pose
            current_pos_right, current_rot_right: (3,), (3,3) -- current right EE pose
        """
        raise NotImplementedError("Implement RobotEnv.get_obs for your actual hardware/simulator.")

    def send_action(self, target_pos_left, target_rot_left, gripper_left,
                     target_pos_right, target_rot_right, gripper_right):
        """Send one step's target end-effector poses + gripper widths to the robot.
        Typically: run IK to get joint targets (cf. deploy_h1.py's H1_2_ArmIK), then
        command the arm/hand controllers.
        """
        raise NotImplementedError("Implement RobotEnv.send_action for your actual hardware/simulator.")


class ObsSaver:
    """Async observation logger -- doesn't block the control loop. Adapted from
    VB-VLA's deploy_scripts/eval_real_bimanual_vb.py::ObsSaver, key names changed to
    match this repo's camera0/camera1/tactile_left_0 etc. convention."""

    def __init__(self, save_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"eval_obs_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.queue = Queue(maxsize=100)
        self.thread = None
        self.running = False
        self.step_count = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def save(self, obs: dict, action: np.ndarray):
        if not self.running:
            return
        try:
            self.queue.put_nowait((self.step_count, obs, action))
            self.step_count += 1
        except Exception:
            pass  # queue full, drop this step rather than block the control loop

    def _worker(self):
        while self.running:
            try:
                step_idx, obs, action = self.queue.get(timeout=1.0)
            except Exception:
                continue
            step_dir = self.save_dir / f"step_{step_idx:06d}"
            step_dir.mkdir(exist_ok=True)
            for key in ("img1", "img2"):
                if key in obs:
                    cv2.imwrite(str(step_dir / f"{key}.jpg"), cv2.cvtColor(obs[key], cv2.COLOR_RGB2BGR))
            if obs.get("tactile_imgs"):
                for key, img in obs["tactile_imgs"].items():
                    cv2.imwrite(str(step_dir / f"{key}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            np.save(step_dir / "obs.npy", obs["obs"])
            np.save(step_dir / "action.npy", action)
            self.queue.task_done()


def main(args):
    yaml_config = yaml.safe_load(open(args.yaml, "r"))
    chunk_size = yaml_config["model"]["chunk_size"]
    use_tactile = yaml_config["model"].get("use_tactile", False)

    temporal_ensembler = ACTTemporalEnsembler(args.temporal_ensembler_alpha, chunk_size) if args.temporal_ensembler else None
    if temporal_ensembler is not None:
        temporal_ensembler.reset()
    action_queue = []  # receding-horizon queue, used when temporal_ensembler is disabled

    model = modeling(yaml_config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    env = RobotEnv()  # TODO: swap in your concrete RobotEnv subclass

    obs_saver = None
    if args.save_obs:
        obs_saver = ObsSaver(args.obs_save_dir)
        obs_saver.start()
        print(f"[ObsSaver] logging to {obs_saver.save_dir}")

    dt = 1.0 / args.control_frequency
    print("Ready. Running at", args.control_frequency, "Hz. Ctrl-C to stop.")

    try:
        while True:
            loop_start = time.time()

            obs = env.get_obs()
            tactile_imgs = obs.get("tactile_imgs") if use_tactile else None

            if len(action_queue) == 0:
                action = predict_action_vitac(
                    model, device, yaml_config,
                    obs["img1"], obs["img2"], obs["obs"],
                    prompt=args.language_prompt,
                    tactile_imgs=tactile_imgs,
                )  # (chunk_size, 20), already denormalized
                action = action.numpy()

                if temporal_ensembler is not None:
                    step_action = temporal_ensembler.update(torch.from_numpy(action).unsqueeze(0))
                    step_action = step_action.squeeze(0).numpy()
                else:
                    # receding horizon: keep the first n_action_select steps of this
                    # chunk, re-infer once they're consumed (same pattern as
                    # deploy_h1.py's `receding_horizon` branch)
                    action_queue = list(action[1: args.select_action])
                    step_action = action[0]
            else:
                step_action = action_queue.pop(0)

            left10, right10 = step_action[:10], step_action[10:]
            target_pos_l, target_rot_l, gripper_l = compose_delta_pose(
                obs["current_pos_left"], obs["current_rot_left"], left10)
            target_pos_r, target_rot_r, gripper_r = compose_delta_pose(
                obs["current_pos_right"], obs["current_rot_right"], right10)

            env.send_action(target_pos_l, target_rot_l, gripper_l,
                             target_pos_r, target_rot_r, gripper_r)

            if obs_saver is not None:
                obs_saver.save(obs, step_action)

            loop_time = time.time() - loop_start
            if loop_time > dt:
                print(f"[warn] loop took {loop_time:.3f}s, slower than target dt={dt:.3f}s")
            time.sleep(max(0.0, dt - loop_time))

    except KeyboardInterrupt:
        print("Interrupted, stopping.")
    finally:
        if obs_saver is not None:
            obs_saver.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="./config/deco_vitac2026_tactile.yaml",
                        help="path to the trained model config; set model.adapter_model_path "
                             "(stage-2 checkpoints) or model.pretrain_model_path (stage-1-only, "
                             "use_tactile=False) to your trained checkpoint")
    parser.add_argument("--language-prompt", type=str, required=True,
                        help="task prompt, embedded live via lang_encoder.FrozenTextEncoder")
    parser.add_argument("--control-frequency", type=float, default=10.0, help="Hz")
    parser.add_argument("--select-action", type=int, default=8,
                        help="steps to consume per inferred chunk before re-inferring, "
                             "when --temporal-ensembler is off")
    parser.add_argument("--temporal-ensembler", action="store_true",
                        help="smooth actions across overlapping chunks instead of "
                             "receding-horizon chunk consumption")
    parser.add_argument("--temporal-ensembler-alpha", type=float, default=0.1)
    parser.add_argument("--save-obs", action="store_true", help="async-log obs/actions for debugging")
    parser.add_argument("--obs-save-dir", type=str, default="./deploy/eval_obs_data")
    args = parser.parse_args()

    main(args)
