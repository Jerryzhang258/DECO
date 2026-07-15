"""
Concrete deco_vitac deployment on the real dual-arm rig used to collect the
black_smash_* datasets -- the same physical hardware as VB-VLA's real_world/
(UvcCamera + arm Controller), just running DECO's flow-matching policy instead of
openpi's pi0/pi0.5, and DECO's own action/obs contract instead of VB-VLA's "quest
pose" one.

Requires VB-VLA on the Python path (--vbvla-root, or set $VBVLA_ROOT).

## Why identity quest_2_ee calibration

VB-VLA's BimanualUmiEnv converts every pose through a `quest_2_ee` matrix -- the
fixed transform from wherever a Meta Quest VR controller was held during their own
teleoperated data collection to the physical end-effector flange (see
Data_collection/vitamin_b_data_collection_pipeline/convert_quest_poses.py and
tools/cali_hand_eye/). DECO's action contract (config/deco_vitac2026_*.yaml's
`action_dim: 20` comment: "per-arm 3D delta-xyz + 6D rotation delta + 1D absolute
gripper width") has no notion of a Quest reference frame at all, so this script
constructs BimanualUmiEnv with IDENTITY quest_2_ee matrices by default -- which makes
`robot{i}_eef_pos`/`rot_axis_angle` and exec_actions' target pose just the raw
end-effector pose in the controller's own frame, unchanged. If the specific
black_smash_* episodes this model trained on *did* go through a Quest-calibrated
collection pipeline after all, pass the real calibration matrices with
--quest-2-ee-left/--quest-2-ee-right (same .npy files deploy_scripts uses) instead.

## The 20-D proprioceptive state

The official obs contract (config comment) is "per-arm 6D relative-to-episode-start
pose + 1D gripper width, x2, + 6D inter-hand relative pose". `_build_obs20` below
implements exactly that from BimanualUmiEnv's raw pose readout and an episode-start
pose captured at the first get_obs() call after reset(). Checked against real logged
frames from KaiyueChen/black_smash_03 (episodes 0-2): the per-arm relative-pose
dims are ~0 at frame_index==0 (as they must be -- current pose equals the episode's
own start pose there) and drift away from 0 over the episode; gripper width stays
roughly constant within an episode (absolute value, not a delta); the inter-hand
relative-pose dims are stable within an episode but differ across episodes
(depends on each episode's initial hand placement) -- all consistent with this
formula. Still worth a final check against a couple of real rollouts on your actual
robot before trusting it fully (rotation convention in particular -- this assumes
axis-angle "rotvec", matching VB-VLA's own pose_util.py convention -- can't be
distinguished from all-zero frame-0 data alone), and use --dry-run until you have.
"""

import os
import sys
import time
import argparse

import numpy as np
import scipy.spatial.transform as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deploy_vitac2026 import RobotEnv, ObsSaver, rotation6d_to_matrix, compose_delta_pose  # noqa: E402
from inference import predict_action_vitac, modeling, ACTTemporalEnsembler  # noqa: E402

import yaml  # noqa: E402
import torch  # noqa: E402


def _add_vbvla_to_path(vbvla_root: str):
    if vbvla_root not in sys.path:
        sys.path.insert(0, vbvla_root)


class RealUmiEnv(RobotEnv):
    """RobotEnv backed by VB-VLA's real_world.BimanualUmiEnv (real arms + UMI
    fisheye/tactile hand cameras)."""

    def __init__(self, cam_path, control_frequency, use_tactile,
                 quest_2_ee_left=None, quest_2_ee_right=None,
                 width_slope=2.041300, width_offset=0.110115, vel_max=0.4,
                 single_arm_mode=False):
        from multiprocessing.managers import SharedMemoryManager
        from real_world.bimanual_umi_env import BimanualUmiEnv

        self.use_tactile = use_tactile
        self.episode_start_pose = None  # set on first get_obs() after (re)start

        self._shm_manager = SharedMemoryManager()
        self._shm_manager.start()
        self.env = BimanualUmiEnv(
            data_type="vitac" if use_tactile else "vision",
            cam_path=cam_path,
            control_frequency=control_frequency,
            obs_image_resolution=(224, 224),
            obs_float32=False,  # keep uint8 HWC -- matches inference.preprocess_vitac's expected input
            camera_obs_horizon=1,
            robot_obs_horizon=1,
            gripper_obs_horizon=1,
            shm_manager=self._shm_manager,
            quest_2_ee_left=quest_2_ee_left if quest_2_ee_left is not None else np.eye(4),
            quest_2_ee_right=quest_2_ee_right if quest_2_ee_right is not None else np.eye(4),
            width_slope=width_slope,
            width_offset=width_offset,
            vel_max=vel_max,
            single_arm_mode=single_arm_mode,
        )
        self.env.start(wait=True)
        print("[RealUmiEnv] waiting for cameras/controller...")
        time.sleep(3.0)

    def reset_episode(self):
        """Call once at the start of each rollout to (re)anchor the relative-pose obs."""
        raw = self.env.get_obs()
        self.episode_start_pose = {
            "left": np.concatenate([raw["robot0_eef_pos"][-1], raw["robot0_eef_rot_axis_angle"][-1]]),
            "right": np.concatenate([raw["robot1_eef_pos"][-1], raw["robot1_eef_rot_axis_angle"][-1]]),
        }

    def _build_obs20(self, raw) -> np.ndarray:
        """See module docstring's correctness warning before trusting this."""
        if self.episode_start_pose is None:
            self.reset_episode()

        pos_l, rot_l = raw["robot0_eef_pos"][-1], raw["robot0_eef_rot_axis_angle"][-1]
        pos_r, rot_r = raw["robot1_eef_pos"][-1], raw["robot1_eef_rot_axis_angle"][-1]
        grip_l, grip_r = raw["robot0_gripper_width"][-1], raw["robot1_gripper_width"][-1]

        start_l = self.episode_start_pose["left"]
        start_r = self.episode_start_pose["right"]
        rel_pos_l = pos_l - start_l[:3]
        rel_rot_l = (st.Rotation.from_rotvec(rot_l) * st.Rotation.from_rotvec(start_l[3:]).inv()).as_rotvec()
        rel_pos_r = pos_r - start_r[:3]
        rel_rot_r = (st.Rotation.from_rotvec(rot_r) * st.Rotation.from_rotvec(start_r[3:]).inv()).as_rotvec()

        inter_hand_pos = pos_r - pos_l
        inter_hand_rot = (st.Rotation.from_rotvec(rot_r) * st.Rotation.from_rotvec(rot_l).inv()).as_rotvec()

        return np.concatenate([
            rel_pos_l, rel_rot_l, [grip_l],
            rel_pos_r, rel_rot_r, [grip_r],
            inter_hand_pos, inter_hand_rot,
        ]).astype(np.float32)

    def get_obs(self) -> dict:
        raw = self.env.get_obs()
        tactile_imgs = None
        if self.use_tactile:
            tactile_imgs = {
                "tactile_left_0": raw["camera0_left_tactile"][-1],
                "tactile_right_0": raw["camera0_right_tactile"][-1],
                "tactile_left_1": raw["camera1_left_tactile"][-1],
                "tactile_right_1": raw["camera1_right_tactile"][-1],
            }
        return dict(
            img1=raw["camera0_rgb"][-1],
            img2=raw["camera1_rgb"][-1],
            tactile_imgs=tactile_imgs,
            obs=self._build_obs20(raw),
            current_pos_left=raw["robot0_eef_pos"][-1],
            current_rot_left=st.Rotation.from_rotvec(raw["robot0_eef_rot_axis_angle"][-1]).as_matrix(),
            current_pos_right=raw["robot1_eef_pos"][-1],
            current_rot_right=st.Rotation.from_rotvec(raw["robot1_eef_rot_axis_angle"][-1]).as_matrix(),
        )

    def send_action(self, target_pos_left, target_rot_left, gripper_left,
                     target_pos_right, target_rot_right, gripper_right):
        rotvec_left = st.Rotation.from_matrix(target_rot_left).as_rotvec()
        rotvec_right = st.Rotation.from_matrix(target_rot_right).as_rotvec()
        action = np.concatenate([
            target_pos_left, rotvec_left, [gripper_left],
            target_pos_right, rotvec_right, [gripper_right],
        ])[None, :]  # exec_actions expects (T, 14)
        # small fixed lead time so the timestamp is in the future when it reaches the controller
        timestamps = np.array([time.time() + 0.1])
        self.env.exec_actions(actions=action, timestamps=timestamps)

    def close(self):
        self.env.stop(wait=True)
        self._shm_manager.shutdown()


def main(args):
    _add_vbvla_to_path(args.vbvla_root)

    yaml_config = yaml.safe_load(open(args.yaml, "r"))
    chunk_size = yaml_config["model"]["chunk_size"]
    use_tactile = yaml_config["model"].get("use_tactile", False)

    temporal_ensembler = ACTTemporalEnsembler(args.temporal_ensembler_alpha, chunk_size) if args.temporal_ensembler else None
    if temporal_ensembler is not None:
        temporal_ensembler.reset()
    action_queue = []

    model = modeling(yaml_config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    quest_2_ee_left = np.load(args.quest_2_ee_left) if args.quest_2_ee_left else None
    quest_2_ee_right = np.load(args.quest_2_ee_right) if args.quest_2_ee_right else None

    env = RealUmiEnv(
        cam_path=args.cam_path,
        control_frequency=args.control_frequency,
        use_tactile=use_tactile,
        quest_2_ee_left=quest_2_ee_left,
        quest_2_ee_right=quest_2_ee_right,
        width_slope=args.width_slope,
        width_offset=args.width_offset,
        vel_max=args.vel_max,
        single_arm_mode=args.single_arm_mode,
    )
    env.reset_episode()

    obs_saver = None
    if args.save_obs:
        obs_saver = ObsSaver(args.obs_save_dir)
        obs_saver.start()
        print(f"[ObsSaver] logging to {obs_saver.save_dir}")

    if args.dry_run:
        print("[dry-run] target poses will be computed and printed, NOT sent to the robot.")

    dt = 1.0 / args.control_frequency
    print(f"Ready. {'DRY RUN, ' if args.dry_run else ''}running at {args.control_frequency} Hz. Ctrl-C to stop.")

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
                ).numpy()

                if temporal_ensembler is not None:
                    step_action = temporal_ensembler.update(torch.from_numpy(action).unsqueeze(0)).squeeze(0).numpy()
                else:
                    action_queue = list(action[1: args.select_action])
                    step_action = action[0]
            else:
                step_action = action_queue.pop(0)

            left10, right10 = step_action[:10], step_action[10:]
            target_pos_l, target_rot_l, gripper_l = compose_delta_pose(
                obs["current_pos_left"], obs["current_rot_left"], left10)
            target_pos_r, target_rot_r, gripper_r = compose_delta_pose(
                obs["current_pos_right"], obs["current_rot_right"], right10)

            if args.dry_run:
                print(f"[dry-run] left target pos={target_pos_l.round(4)} gripper={gripper_l:.4f} | "
                      f"right target pos={target_pos_r.round(4)} gripper={gripper_r:.4f}")
            else:
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
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbvla-root", type=str, default=os.environ.get("VBVLA_ROOT", "/Users/zhangrongxuan/VB-VLA"),
                        help="path to the VB-VLA repo (provides real_world/BimanualUmiEnv, utils/pose utilities)")
    parser.add_argument("--yaml", type=str, default="./config/deco_vitac2026_tactile.yaml")
    parser.add_argument("--language-prompt", type=str, required=True)
    parser.add_argument("--cam-path", type=str, nargs="+", default=["/dev/video0", "/dev/video2"],
                         help="camera0 (left hand) then camera1 (right hand)")
    parser.add_argument("--control-frequency", type=float, default=10.0)
    parser.add_argument("--select-action", type=int, default=8)
    parser.add_argument("--temporal-ensembler", action="store_true")
    parser.add_argument("--temporal-ensembler-alpha", type=float, default=0.1)
    parser.add_argument("--quest-2-ee-left", type=str, default=None,
                        help="only needed if black_smash_* was collected through a Quest-calibrated "
                             "pipeline -- see module docstring. Defaults to identity (no transform).")
    parser.add_argument("--quest-2-ee-right", type=str, default=None)
    parser.add_argument("--width-slope", type=float, default=2.041300)
    parser.add_argument("--width-offset", type=float, default=0.110115)
    parser.add_argument("--vel-max", type=float, default=0.4)
    parser.add_argument("--single-arm-mode", action="store_true")
    parser.add_argument("--save-obs", action="store_true")
    parser.add_argument("--obs-save-dir", type=str, default="./deploy/eval_obs_data")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and print target poses without sending them to the robot -- "
                             "use this until the obs20/quest-frame assumptions above are verified")
    args = parser.parse_args()

    main(args)
