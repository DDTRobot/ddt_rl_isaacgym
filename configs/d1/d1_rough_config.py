from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# config
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot

class D1Rough(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.hip_joint_indices = [0, 4, 8, 12]
        self.foot_joint_indices = [3, 7, 11, 15]
        #phase related
        self.phase = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.phase_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.frequency = 2.
        
        self.trot_gait = torch.zeros(1, 4, dtype=torch.float, device=self.device,requires_grad=False)
        self.trot_gait[:,0] = torch.pi
        self.trot_gait[:,-1] = torch.pi

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        # actions = self.reindex(actions)
        actions = actions.to(self.device)

        self.global_counter += 1   
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_pos[:, self.foot_joint_indices]  = 0  # zero position of wheels 
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.cost_buf,self.reset_buf, self.extras
    
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._process_phase()
    
    def _process_phase(self):
        """update phase value for all actor"""
        self.phase_time = torch.fmod(self.frequency*self.dt + self.phase_time,1.0)
        self.phase = 2*torch.pi*self.phase_time+self.trot_gait
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.phase[env_ids,:] = 0
        self.phase_time[env_ids,:] = 0

    
    def compute_observations(self):
        self.dof_pos[:,[3,7,11,15]] = 0
        obs_buf =torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                            self.base_ang_vel  * self.obs_scales.ang_vel,
                            self.projected_gravity,
                            self.commands[:, :3] * self.commands_scale,
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                            (self.dof_vel * self.obs_scales.dof_vel),
                            # torch.norm(self.commands[:, :3] * self.commands_scale,dim=-1,keepdim=True)*torch.sin(self.phase),
                            # torch.norm(self.commands[:, :3] * self.commands_scale,dim=-1,keepdim=True)*torch.cos(self.phase),
                            #self.reindex_feet(self.contact_filt.float()-0.5),
                            # self.reindex(self.action_history_buf[:,-1])),dim=-1)
                            self.action_history_buf[:,-1]),dim=-1)

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.zeros(3),
                               torch.ones(3) * noise_scales.ang_vel * noise_level,
                               torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.zeros(3),
                               torch.ones(
                                   16) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   16) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                            #    torch.zeros(4),
                            #    torch.zeros(4),
                               #torch.ones(4) * noise_scales.contact_states * noise_level,
                               #torch.zeros(4),
                               torch.zeros(self.num_actions),
                               ), dim=0)
        
        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * noise_vec.to(self.device)

        priv_latent = torch.cat((
            #self.base_lin_vel * self.obs_scales.lin_vel,
            self.contact_filt.float()-0.5,
            self.randomized_lag_tensor,
            #self.base_ang_vel  * self.obs_scales.ang_vel,
            # self.base_lin_vel * self.obs_scales.lin_vel,
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.restitution_coeffs_tensor,
            self.motor_strength, 
            self.kp_factor,
            self.kd_factor,
            torch.sin(self.phase)), dim=-1)
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            #priv_latent = torch.cat([priv_latent,self.feet_local_heights],dim=-1)
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)*self.obs_scales.height_measurements
            self.obs_buf = torch.cat([obs_buf, heights, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        # update buffer
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

        if self.cfg.terrain.include_act_obs_pair_buf:
            # add to full observation history and action history to obs
            pure_obs_hist = self.obs_history_buf[:,:,:-self.num_actions].reshape(self.num_envs,-1)
            act_hist = self.action_history_buf.view(self.num_envs,-1)
            self.obs_buf = torch.cat([self.obs_buf,pure_obs_hist,act_hist], dim=-1)
    
    def _compute_torques(self, actions):

        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # 如果使用滤波器，则对动作进行滤波
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)

        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target - self.dof_pos) - self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]                
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices]
                - self.kd_factor[:,self.foot_joint_indices] *self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        else: 
            raise NameError(f"Unknown controller type: {control_type}")
        torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    #------------ reward functions----------------
    def _reward_foot_mirror(self):
        diff1 = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[12,13,14]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:,[4,5,6]] - self.dof_pos[:,[8,9,10]]),dim=-1)
        # diff3 = torch.sum(torch.square(self.dof_vel[:,3] - self.dof_vel[:,15]),dim=-1)
        # diff4 = torch.sum(torch.square(self.dof_vel[:,7] - self.dof_vel[:,11]),dim=-1)
        return 0.5*(diff1 + diff2)

    def _reward_phase_contact(self):
        local_footvel = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            local_footvel[:, i, :] = quat_rotate_inverse(self.base_quat, self.feet_vel[:, i, :])
        mean_local_footvel = torch.mean(local_footvel, dim=1)
        weight_com_vel = 0.5 * mean_local_footvel[:, 0] + 0.5 * self.base_lin_vel[:, 0]
        contact_goal = 1.*(torch.sin(self.phase) > 0.0)
        return torch.mean(torch.abs(1.*self.contact_filt - contact_goal),dim=1) * torch.logical_or(torch.abs(self.commands[:, 2]) > 0.2, torch.abs(self.commands[:, 0] - weight_com_vel) > 0.15)
    
    def _reward_com_feet_contact(self):
        com = self.root_states[:, 0:2]
        feet_com = (self.feet_pos[:, 0, 0:2] + self.feet_pos[:, 1, 0:2] + self.feet_pos[:, 2, 0:2] + self.feet_pos[:, 3, 0:2])/4
        feet_com_contact = self.feet_pos[:, 0, 0:2] * self.contact_filt[:, 0].unsqueeze(1) + self.feet_pos[:, 1, 0:2] * self.contact_filt[:, 1].unsqueeze(1) + self.feet_pos[:, 2, 0:2] * self.contact_filt[:, 2].unsqueeze(1) + self.feet_pos[:, 3, 0:2] * self.contact_filt[:, 3].unsqueeze(1)
        feet_contact_num = self.contact_filt[:, 0].float() + self.contact_filt[:, 1].float() + self.contact_filt[:, 2].float() + self.contact_filt[:, 3].float()
        feet_com = torch.where(feet_contact_num.unsqueeze(1) > 0, feet_com_contact / feet_contact_num.unsqueeze(1), feet_com)
        err = torch.sum(torch.square(com - feet_com), dim=1)
        return torch.exp(-err/0.25)

    def _reward_too_much_air(self):
        feet_contact_num = self.contact_filt[:, 0].float() + self.contact_filt[:, 1].float() + self.contact_filt[:, 2].float() + self.contact_filt[:, 3].float()
        return feet_contact_num < 1.9
    
    def _reward_square_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(torch.square((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.)), dim=1)

class D1RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096

        n_scan = 187
        n_priv_latent =  63
        n_proprio = 60 #
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 16
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.49] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 1.0,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 1.0,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,   # [rad]
            'RR_calf_joint': -1.5,   # [rad]

            'FL_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 40.,
                     'thigh': 40.,
                     'calf': 40.,
                     'foot': 10.}  # [N*m/rad]
        damping = {'hip': 1.0,
                   'thigh': 1.0,
                   'calf': 1.0,
                   'foot': 0.5}     #  [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/ddt_ros2_control/urdfs/d1_description/urdf/robot.urdf'
        foot_name = "foot"
        name = "d1"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            dof_pos_limits = 0.0
            feet_air_time = 0.0
            powers = -2e-5
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            action_rate = -0.01
            action_smoothness = -0.002
            orientation = -0.2
            
            foot_mirror = -0.05
            collision = -1
            base_height = -10.0
            stumble = -0.05
            com_feet_contact = 0.5
            too_much_air = -2
            phase_contact = -1
            square_feet_contact_forces = -1e-5

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.49
        max_contact_force = 700.  # forces above this value are penalized

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        #friction_range = [0.2, 2.75]
        friction_range = [0.2, 1.25]
        randomize_restitution = False
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        #added_mass_range = [-1., 3.]
        added_mass_range = [-1, 2.]
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1

        randomize_motor = True
        motor_strength_range = [0.9, 1.1]

        randomize_kpkd = True
        kp_range = [0.9,1.1]
        kd_range = [0.9,1.1]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        # randomize_initial_joint_pos = True
        # initial_joint_pos_range = [0.5, 1.5]
    
    class costs:
        num_costs = 3
        class scales:
            pos_limit = 0.1
            torque_limit = 0.1
            dof_vel_limits = 0.1

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        include_act_obs_pair_buf = False

class D1RoughCfg_Play( D1RoughCfg ):
    class env(D1RoughCfg.env):
        num_envs = 1
    class terrain(D1RoughCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        num_rows = 5
        num_cols = 5
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0, 0, 1, 0, 0, 0, 0]
        curriculum = False

    class noise( D1RoughCfg.noise ):
        add_noise = False
    class control ( D1RoughCfg.control ):
        use_filter = True

    class domain_rand( D1RoughCfg.domain_rand ):
        push_robots = False
        randomize_friction = False
        randomize_base_com = False
        randomize_base_mass = False
        randomize_motor = False
        randomize_lag_timesteps = False
        randomize_friction = False
        randomize_restitution = False
        disturbance = False
        randomize_kpkd = False

class D1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 3

        teacher_act = True
        imi_flag = True
      
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'd1_rough'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 6500
        num_steps_per_env = 24
        resume = False
        resume_path = ''
 

  
