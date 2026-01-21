import os
import numpy as np

def get_unique_filename(base_path, ext=".mp4"):
    """중복된 파일명이 존재하면 숫자를 증가하여 새로운 경로를 반환"""
    if not base_path.endswith(ext):
        base_path += ext  # 확장자 자동 추가

    file_name, file_ext = os.path.splitext(base_path)  # 파일명과 확장자 분리
    count = 0
    new_path = f"{file_name}-episode-0"+file_ext

    while os.path.exists(new_path):  # 파일 존재 여부 확인
        new_path = f"{file_name}{count}-episode-0{file_ext}"
        count += 1


    return f"rl-video{count-1}", new_path

def get_data_from_info(info):
    # Action info
    action = np.array([_info['action'] for _info in info])

    # Status info
    stat_init_rpy = np.array([_info['init_rpy'] for _info in info])
    stat_init_com = np.array([_info['init_com'] for _info in info])
    stat_xy_vel = np.array([[_info['x_velocity'], _info['y_velocity']] for _info in info])
    stat_yaw_vel = np.array([_info['yaw_velocity'] for _info in info])
    stat_quat = np.array([_info['head_quat'] for _info in info])
    stat_ang_vel = np.array([_info['head_ang_vel'] for _info in info])
    stat_lin_acc = np.array([_info['head_lin_acc'] for _info in info])
    stat_motion_vector = np.array([_info['motion_vector'] for _info in info])
    stat_com_pos = np.array([_info['com_pos'] for _info in info])
    stat_com_ypr = np.array([_info['com_ypr'] for _info in info])
    stat_step_ypr = np.array([_info['step_ypr'] for _info in info])
    stat_reward_func_orientation = np.array([_info['reward_func_orientation'] for _info in info])
    

    # Rew info
    rew_linear_movement = np.array([_info['reward_linear_movement'] for _info in info])
    reward_angular_movement = np.array([_info['reward_angular_movement'] for _info in info])
    reward_efficiency = np.array([_info['reward_efficiency'] for _info in info])
    reward_healthy = np.array([_info['reward_healthy'] for _info in info])
    cost_ctrl = np.array([_info['cost_ctrl'] for _info in info])
    cost_unhealthy = np.array([_info['cost_unhealthy'] for _info in info])
    cost_orientation = np.array([_info['cost_orientation'] for _info in info])
    cost_yaw_vel = np.array([_info['cost_yaw_vel'] for _info in info])
    cost_proj_dist = np.array([_info['cost_proj_dist'] for _info in info])
    direction_similarity = np.array([_info['direction_similarity'] for _info in info])
    rotation_alignment = np.array([_info['rotation_alignment'] for _info in info])
    vel_orientation = np.array([_info['velocity_theta'] for _info in info])

    # Input info
    input_joy = np.array([_info['joy_input'] for _info in info])
    gait_param = np.array([_info['gait_params'] for _info in info])

    data_dict = {
        'action': action,
        'stat_init_rpy': stat_init_rpy,
        'stat_init_com': stat_init_com,
        'stat_xy_vel': stat_xy_vel,
        'stat_yaw_vel': stat_yaw_vel,
        'stat_quat': stat_quat,
        'stat_ang_vel': stat_ang_vel,
        'stat_lin_acc': stat_lin_acc,
        'stat_motion_vector': stat_motion_vector,
        'stat_com_pos': stat_com_pos,
        'stat_com_ypr': stat_com_ypr,
        'stat_com_r_ypr':stat_reward_func_orientation,
        'stat_step_ypr': stat_step_ypr,

        'rew_linear_movement': rew_linear_movement,
        'reward_angular_movement': reward_angular_movement,
        'reward_efficiency': reward_efficiency,
        'reward_healthy': reward_healthy,
        'cost_ctrl': cost_ctrl,
        'cost_unhealthy': cost_unhealthy,
        'cost_orientation': cost_orientation,
        'cost_yaw_vel': cost_yaw_vel,
        'cost_proj_dist': cost_proj_dist,
        'direction_similarity': direction_similarity,
        'rotation_alignment': rotation_alignment,
        'vel_orientation': vel_orientation,

        'input_joy': input_joy,
        'gait_param': gait_param,
    }
    
    return data_dict