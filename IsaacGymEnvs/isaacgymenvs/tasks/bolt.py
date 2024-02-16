# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse
from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict
import wandb


class Bolt(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["balance_speed"] = self.cfg["env"]["learn"]["balanceSpeedRewardScale"]
        self.rew_scales["balance_rotation"] = self.cfg["env"]["learn"]["balanceRotationRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["dof_vel"] = self.cfg["env"]["learn"]["dofVelocityRewardScale"]
        self.rew_scales["dof_acc"] = self.cfg["env"]["learn"]["dofAccelerationRewardScale"]
        self.rew_scales["action_change"] = self.cfg["env"]["learn"]["actionChangeRewardScale"]
        self.rew_scales["slip"] = self.cfg["env"]["learn"]["slipRewardScale"]
        self.rew_scales["joint_limit"] = self.cfg["env"]["learn"]["jointLimitRewardScale"]
        #self.rew_scales["base_flat"] = self.cfg["env"]["learn"]["baseFlatRewardScale"]
        self.rew_scales["maxHeight"] = self.cfg["env"]["learn"]["maxFootHeightReward"]
        self.rew_scales["clearance"] = self.cfg["env"]["learn"]["clearanceRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 30
        self.cfg["env"]["numActions"] = 6

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.knee_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.knee_indices, 0:3]

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]

        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.vertical_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros_like(self.actions)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # IMPORTANT: By default (and unchangable), rl_games steps returns the following names:
        # self.self.obs_dict, self.rew_buf, self.reset_buf, self.extras (you may check on the VecTask class, step method)
        # These extras are used by the Observers (here they call it info... god knows why) to compute metrics
        # So, if you want to access the Observers (such as wandb/rlgpu observer), you must add them to the self.extras dict
        self.extras = {}
        reward_keys = [
            "lin_vel_xy",
            "ang_vel_z",
            "balance_speed",
            "balance_rotation",
            "torque",
            "dof_vel",
            "dof_acc",
            "action_change",
            "joint_limit",
            "slip",
            "clearance",
            #"base_flat",
            #"air_time",
            "total_reward"
        ]
        self.rewards_episode = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) for key in reward_keys
        }

        self.extras["episode_cumulative"] = self.rewards_episode

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg["env"]["learn"]["addNoise"]
        noise_level = self.cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self.cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self.cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self.cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:18] = self.cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[18:24] = self.cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[24:30] = 0. # previous actions
        return noise_vec


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/bolt_new.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = False
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        bolt_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(bolt_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(bolt_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(bolt_asset)
        self.dof_names = self.gym.get_asset_dof_names(bolt_asset)
        extremity_name = "LOWER_LEG" if asset_options.collapse_fixed_joints else "FOOT"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "LOWER_LEG" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        shoulder_names = [s for s in body_names if "SHOULDER" in s]
        self.shoulder_indices = torch.zeros(len(shoulder_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = torch.zeros(1, dtype=torch.long, device=self.device, requires_grad=False)
        base_name = "base_link"

        dof_props = self.gym.get_asset_dof_properties(bolt_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.bolt_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            bolt_handle = self.gym.create_actor(env_ptr, bolt_asset, start_pose, "bolt", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, bolt_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, bolt_handle)
            self.envs.append(env_ptr)
            self.bolt_handles.append(bolt_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bolt_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bolt_handles[0], knee_names[i])
        for i in range(len(shoulder_names)):
            self.shoulder_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bolt_handles[0], shoulder_names[i])

        self.base_index = torch.tensor(self.gym.find_actor_rigid_body_handle(self.envs[0], self.bolt_handles[0], base_name))


    def pre_physics_step(self, actions):
        self.last_actions[:] = self.actions[:]
        self.actions = actions.clone().to(self.device)
        # TODO: Is action_scale necessary? It's default to 0.5 in the config file
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.last_dof_vel[:] = self.dof_vel[:]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.knee_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.knee_indices, 0:3]
        self.compute_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # self.compute_reward(self.actions)
        self.compute_bolt_reward(self.actions)


    def compute_bolt_reward(self, actions):
        base_quat = self.root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])

        # Velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.rew_scales["ang_vel_z"]

        # Keep balance
        rew_balance_speed = torch.square(base_lin_vel[:, 2]) * self.rew_scales["balance_speed"]
        rew_balance_rotation = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1) * self.rew_scales["balance_rotation"]
        
        # Smooth motion rewards
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]
        rew_dof_vel = torch.sum(torch.square(self.dof_vel), dim=1) * self.rew_scales["dof_vel"]
        rew_dof_acc = torch.sum(torch.square(self.dof_vel - self.last_dof_vel), dim=1) * self.rew_scales["dof_acc"] / (self.dt ** 2)
        rew_action_change = torch.sum(torch.square(self.actions - self.last_actions), dim=1) * self.rew_scales["action_change"]

        # Penalty on difference between current position and initial position
        rew_joint_limit = torch.exp(-torch.norm(self.dof_pos - self.default_dof_pos, dim=1)) * self.rew_scales["joint_limit"]

        # foot slip penalty (solo 12 article)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=2) > 1
        rew_slip = torch.sum(contact * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim = 2)), dim = 1) * self.rew_scales["slip"]

        # foot clearance penalty (solo 12 article)
        rew_clearance = torch.sum(torch.square(self.foot_positions[:, :, 2] - self.rew_scales["maxHeight"]) * torch.sqrt(torch.norm(self.foot_velocities[:, :, :2], dim = 2)), dim = 1) * self.rew_scales["clearance"]

        # power loss penalty (solo 12 article)
        #rew_power_loss = torch.sum() * rew_scales["power_loss"]

        # action smoothness penalty (solo 12 article) + heuristic based
        #rew_smoothness = torch.square() * rew_scales["smoothness1"] + torch.square() * rew_scale["smoothness2"]

        # reward for air time (solo 12 article)
        # 1/(1 + exp(-t/0.25)
        # rew_air_time = 0.0025/(1 + torch.exp(-self.progress_buf/0.5)) #progress_buf == episode_lengths

        # Base stay stable
        # projected_vertical = quat_rotate(base_quat, self.vertical_vec)
        # rew_base_flat = torch.square(torch.norm(projected_vertical - self.vertical_vec, dim=1)) * self.rew_scales["base_flat"]

        total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_balance_speed + rew_balance_rotation + rew_torque + rew_dof_vel + rew_dof_acc + rew_action_change + rew_joint_limit + rew_slip + rew_clearance # + rew_air_time + rew_base_flat
        total_reward = torch.clip(total_reward, 0., None)

        # reset agents
        reset_kneeling = torch.any(self.knee_positions[:, :, 2] - self.foot_positions[:, :, 2] < 0.1, dim=1)
        # reset1 = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        # reset2 = torch.any(torch.norm(self.contact_forces[:, self.shoulder_indices, :], dim=2) > 1., dim=1)
        # reset3 = torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1., dim=1)
        time_out = self.progress_buf >= self.max_episode_length - 1  # no terminal reward for time-outs
        reset = reset_kneeling | time_out

        # log metrics
        all_rewards = [
            (rew_ang_vel_z, "ang_vel_z"),
            (rew_lin_vel_xy, "lin_vel_xy"),
            (rew_balance_speed, "balance_speed"),
            (rew_balance_rotation, "balance_rotation"),
            (rew_torque, "torque"),
            (rew_dof_vel, "dof_vel"),
            (rew_dof_acc, "dof_acc"),
            (rew_action_change, "action_change"),
            (rew_joint_limit, "joint_limit"),
            (rew_slip, "slip"),
            #(rew_base_flat, "base_flat"),
            #(rew_air_time, "air_time"),
            (rew_clearance, "clearance"),
            (total_reward, "total_reward"),
        ]

        episode_cumulative = dict()
        for rew, name in all_rewards:
            self.rewards_episode[name] += rew
            episode_cumulative[name] = rew
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        self.rew_buf[:] = total_reward
        self.reset_buf[:] = reset

    # def compute_reward(self, actions):
    #     self.rew_buf[:], self.reset_buf[:] = compute_bolt_reward(
    #         # tensors
    #         self.root_states,
    #         self.commands,
    #         self.torques,
    #         self.contact_forces,
    #         self.shoulder_indices,
    #         self.knee_indices,
    #         self.progress_buf,
    #         # Dict
    #         self.rew_scales,
    #         # other
    #         self.base_index,
    #         self.max_episode_length,
    #         self.foot_positions, 
    #         self.foot_velocities,
    #         self.feet_indices,
    #     )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.obs_buf[:] = compute_bolt_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        for key in self.rewards_episode.keys():
            self.rewards_episode[key][env_ids] = 0.

        

    

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_bolt_reward(
    # tensors
    root_states,
    commands,
    torques,
    contact_forces,
    shoulder_indices,
    knee_indices,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    base_index,
    max_episode_length,
    foot_positions,
    foot_velocities,
    feet_indices,
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    #print(f"feet_indices: {type(feet_indices)}")
    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]


    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]


    # foot slip penalty (solo 12 article)
    contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1
    rew_slip = torch.sum(contact * torch.square(torch.norm(foot_velocities[:, :, :2], dim = 2)), dim = 1) * rew_scales["slip"]

    #keep balance r = -0.015*(vitesse_rot_base_x²+vitesse_rot_base_y²)
    rew_balance = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1) * rew_scales["balance_rotation"]

    # foot clearance penalty (solo 12 article)
    rew_clearance = torch.sum(torch.square(foot_positions[:, :, 2] - rew_scales["maxHeight"]) * torch.sqrt(torch.norm(foot_velocities[:, :, :2], dim = 2)), dim = 1) * rew_scales["clearance"]

    # bipedal stability penalty (thx to the gravity vector at the center of mass)
    #rew_stability = 

    # power loss penalty (solo 12 article)
    #rew_power_loss = torch.sum() * rew_scales["power_loss"]

    # action smoothness penalty (solo 12 article) + heuristic based
    #rew_smoothness = torch.square() * rew_scales["smoothness1"] + torch.square() * rew_scale["smoothness2"]

    # reward for air time (solo 12 article)
    # 1/(1 + exp(-t/0.25)
    #print(episode_lengths)
    rew_air_time = 0.0025/(1 + torch.exp(-episode_lengths/0.5))
    #print(rew_air_time)
    # penalties from anymal_terrain.py

    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_balance + rew_slip + rew_clearance + rew_air_time
    total_reward = torch.clip(total_reward, 0., None)
    # reset agents
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, shoulder_indices, :], dim=2) > 1., dim=1)
    #reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out

    return total_reward.detach(), reset


@torch.jit.script
def compute_bolt_observations(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions
                     ), dim=-1)

    return obs
