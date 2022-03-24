import argparse
import json
import os
import pickle
import sys
from unicodedata import category
from scipy.spatial.transform import Rotation as R
sys.path.append('/home/harry/discriminative_embeddings')

import numpy as np
from tqdm import trange
from utils import get_pointcloud
import shutil
import trimesh
import spherical_sampling
from model import Model
from sim import PybulletSim
from utils import project_pts_to_2d

sys.path.append('/home/harry/discriminative_embeddings')
from part_embedding.datasets.pm.pm_raw import parse_urdf
from part_embedding.datasets.calc_art import compute_new_points

def create_train_envs():
    train_data = get_train_instances(open('mobility_dataset/split-full.json'))
    val_data = get_val_instances(open('mobility_dataset/split-full.json'))
    test_data = get_test_instances(open('mobility_dataset/split-full.json'))
    sim = PybulletSim(False, 0.18)
    for cat in train_data.keys():
        inst_id = train_data[cat][0]
        observation = sim.reset(scene_state=None, category_type='train', instance_type='train', category_name=cat, instance_id=inst_id)
        pcd = get_pointcloud(observation['image'][:, :, -1], observation['image'], sim.segmentation_mask, sim._scene_cam_intrinsics, sim.cam_pose_matrix)[0]
        pcd, segmasks = post_process_pcd(sim, pcd)
        dataset_path = os.path.expanduser("~/umpnet/mobility_dataset")
        obj_urdf = os.path.join(dataset_path, cat, inst_id, "mobility.urdf")
        pm_obj = parse_urdf(obj_urdf)
        flow_all = {}
        for link in segmasks.keys(): 
            flow = transform_pcd(sim, pcd, segmasks[link], pm_obj.get_chain(link), 0.1)
            flow_all[link] = flow
            # scene = trimesh.Scene([trimesh.points.PointCloud(pcd), trimesh.points.PointCloud(pcd+flow, colors=(255,0,0))])
            # scene.show()

def create_env_by_id(obj_id):
    train_data = get_train_instances(open('mobility_dataset/split-full.json'))
    val_data = get_val_instances(open('mobility_dataset/split-full.json'))
    test_data = get_test_instances(open('mobility_dataset/split-full.json'))
    sim = PybulletSim(False, 0.18)
    for cat in train_data.keys():
        if obj_id in train_data[cat]:
            category_type = 'train'
            instance_type = 'train'
    for cat in val_data.keys():
        if obj_id in val_data[cat]:
            category_type = 'train'
            instance_type = 'test'
    for cat in test_data.keys():
        if obj_id in test_data[cat]:
            category_type = 'test'
            instance_type = 'test'

    observation = sim.reset(scene_state=None, category_type=category_type, instance_type=instance_type, category_name=cat, instance_id=obj_id)
    pcd = get_pointcloud(observation['image'][:, :, -1], observation['image'], sim.segmentation_mask, sim._scene_cam_intrinsics, sim.cam_pose_matrix)[0]
    pcd, segmasks = post_process_pcd(sim, pcd)
    dataset_path = os.path.expanduser("~/umpnet/mobility_dataset")
    obj_urdf = os.path.join(dataset_path, cat, obj_id, "mobility.urdf")
    pm_obj = parse_urdf(obj_urdf)
    flow_all = {}
    for link in segmasks.keys(): 
        flow = transform_pcd(sim, pcd, segmasks[link], pm_obj.get_chain(link), 0.1)
        flow_all[link] = flow


def get_link_name(sim, joint_id):
    """ Get link name based on PyBullet joint id"""
    joint_info = sim.bc.getJointInfo(sim.body_id, joint_id)
    return joint_info[12].decode('gb2312')

def transform_pcd(sim, pcd, seg, chain, magnitude):

    # Get the seg pcd.
    seg_pcd = pcd[np.where(seg)[0], :]
    org_config = np.zeros(len(chain))
    target_config = np.zeros(len(chain))
    target_config[-1] = magnitude
    T_world_base = R.from_quat(sim.base_orientation).as_matrix()
    T_world_base = np.hstack([T_world_base, np.array([0, 0.5*sim.scaling, 0]).reshape(3, 1)])
    T_world_base = np.vstack([T_world_base, [0, 0, 0, 1]])

    p_world_flowedpts = compute_new_points(
        seg_pcd, T_world_base, chain, org_config, target_config
    )

    flow_local = p_world_flowedpts - seg_pcd
    flow = np.zeros_like(pcd)
    flow[np.where(seg)[0], :] = flow_local

    return flow


def post_process_pcd(sim, pcd):
    downsample = pcd[:, 2]>1e-2
    segmasks = {}
    filter = None
    for movable_joint in sim.joint_states.keys():
        segmask = np.zeros_like(sim.link_id_pts)
        segmask[np.where(sim.link_id_pts == movable_joint)] = 1
        segmask = segmask[downsample]
        segmask = segmask.astype('bool')
        if filter is None:
            filter = np.random.permutation(np.arange(segmask.shape[0]))[:1200]
        segmask = segmask[filter]
        link_name = get_link_name(sim, movable_joint)
        segmasks[link_name] = segmask
    pcd = pcd[downsample][filter]
    xyz = pcd - pcd.mean(axis=-2)
    scale = (1 / np.abs(xyz).max()) * 0.999999
    xyz = xyz * scale
    return xyz, segmasks


def get_train_instances(split_path):
    full_split = json.load(split_path)
    train_data = {}
    for cat in full_split['train'].keys():
        if full_split['train'][cat]['train'] and cat != 'Toy':
            train_data[cat] = full_split['train'][cat]['train']
    return train_data

def get_val_instances(split_path):
    full_split = json.load(split_path)
    val_data = {}
    for cat in full_split['train'].keys():
        if full_split['train'][cat]['test'] and cat != 'Toy':
            val_data[cat] = full_split['train'][cat]['test']
    return val_data

def get_test_instances(split_path):
    full_split = json.load(split_path)
    test_data = {}
    for cat in full_split['test'].keys():
        if full_split['test'][cat]['test']:
            test_data[cat] = full_split['test'][cat]['test']
    return test_data

if __name__ == '__main__':
    create_train_envs()
