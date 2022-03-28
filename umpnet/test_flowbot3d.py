import argparse
import json
import os
import pickle
import sys

sys.path.append("/home/harry/discriminative_embeddings")

import numpy as np
import torch
from tqdm import trange
from part_embedding.flow_prediction.latest_models import load_model
from part_embedding.flow_prediction.animate import FlowNetAnimation
from umpnet.utils import get_pointcloud
import shutil

from umpnet.sim import PybulletSim
from umpnet.utils import project_pts_to_2d

step_num_dict = {
    "Refrigerator": 12,
    "FoldingChair": 8,
    "Laptop": 12,
    "Stapler": 15,
    "TrashCan": 9,
    "Microwave": 8,
    "Toilet": 7,
    "Window": 6,
    "StorageFurniture": 9,
    "Switch": 7,
    "Kettle": 3,
    "Toy": 10,
    "Box": 10,
    "Phone": 12,
    "Dishwasher": 10,
    "Safe": 10,
    "Oven": 9,
    "WashingMachine": 9,
    "Table": 7,
    "KitchenPot": 3,
    "Bucket": 13,
    "Door": 10,
}


def main():
    model_name = "flowbot-in-ump"
    mobility_path = "mobility_dataset"
    split_file = "split-full.json"
    split_meta = json.load(open(os.path.join(mobility_path, split_file), "r"))

    # Load model
    device = torch.device(f"cuda:0")
    model = load_model(model_name, False)
    model = model.to("cuda")

    print("==> FlowBot3D model loaded")
    model.eval()
    torch.set_grad_enabled(False)

    pool_list = list()
    for category_type in ["train", "test"]:
        for category_name in split_meta[category_type].keys():
            # if category_name not in ['Refrigerator', 'FoldingChair', 'Laptop', 'Stapler']:
            #     print('Skipping Category...')
            #     continue
            instance_type = "test"
            pool_list.append((category_type, category_name, instance_type))

    for category_type, category_name, instance_type in pool_list:
        run_test(model, category_type, category_name, instance_type)


def predict_flow_gtmask(sim, observation, model, reverse):
    pcd = get_pointcloud(
        observation["image"][:, :, -1],
        observation["image"],
        sim.segmentation_mask,
        sim._scene_cam_intrinsics,
        sim.cam_pose_matrix,
    )[0]
    downsample = pcd[:, 2] > 1e-2
    segmask = np.zeros_like(sim.link_id_pts)
    segmask[np.where(sim.link_id_pts == sim.selected_joint)] = 1
    segmask = segmask[downsample]
    segmask = segmask.astype("bool")

    filter = np.random.permutation(np.arange(segmask.shape[0]))[:1200]

    tracker = 0
    while not segmask[filter].any():
        print("Part Points Ignored During Sampling...")
        print("RESAMPLING")
        filter = np.random.permutation(np.arange(segmask.shape[0]))[:1200]
        tracker += 1
        if tracker > 10:
            segmask = np.ones_like(segmask).astype("bool")
            break

    segmask = segmask[filter]
    pcd = pcd[downsample][filter]
    xyz = pcd - pcd.mean(axis=-2)
    scale = (1 / np.abs(xyz).max()) * 0.999999
    xyz = xyz * scale
    # xyz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])@xyz.T
    # xyz = xyz.T

    pred_flow = model.predict(
        torch.from_numpy(xyz).to("cuda"),
        torch.from_numpy(
            np.ones(
                xyz.shape[0],
            )
        ).float(),
    )
    pred_flow = pred_flow.cpu().numpy()
    # pred_flow = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T@pred_flow.T
    # pred_flow = pred_flow.T
    pred_flow = pred_flow[segmask]
    if reverse:
        pred_flow = -pred_flow
    return pred_flow, segmask, pcd, xyz


def predict_flow_gtmask_after(sim, observation, model, reverse):

    pcd = get_pointcloud(
        observation["image"][:, :, -1],
        observation["image"],
        sim.segmentation_mask,
        sim._scene_cam_intrinsics,
        sim.cam_pose_matrix,
    )[0]

    # Delete gripper points
    pcd = np.delete(pcd, np.where(sim.body_id_pts == sim._suction_gripper), 0)
    # Delete ground points
    downsample = pcd[:, 2] > 1e-2

    # Segment out target link
    segmask = np.zeros_like(sim.link_id_pts)
    segmask[np.where(sim.link_id_pts == sim.selected_joint)] = 1
    segmask = np.delete(segmask, np.where(sim.body_id_pts == sim._suction_gripper), 0)
    segmask = segmask[downsample]
    segmask = segmask.astype("bool")

    # Downsample to 1200
    filter = np.random.permutation(np.arange(segmask.shape[0]))[:1200]

    tracker = 0
    while not segmask[filter].any():
        print("Part Points Ignored During Sampling...")
        print("RESAMPLING")
        filter = np.random.permutation(np.arange(segmask.shape[0]))[:1200]
        tracker += 1
        if tracker > 10:
            segmask = np.ones_like(segmask).astype("bool")
            break

    segmask = segmask[filter]
    pcd = pcd[downsample][filter]
    xyz = pcd - pcd.mean(axis=-2)
    scale = (1 / np.abs(xyz).max()) * 0.999999
    xyz = xyz * scale
    # xyz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])@xyz.T
    # xyz = xyz.T

    pred_flow = model.predict(
        torch.from_numpy(xyz).to("cuda"),
        torch.from_numpy(
            np.ones(
                xyz.shape[0],
            )
        ).float(),
    )
    pred_flow = pred_flow.cpu().numpy()
    # pred_flow = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T@pred_flow.T
    # pred_flow = pred_flow.T
    pred_flow = pred_flow[segmask]
    if reverse:
        pred_flow = -pred_flow
    return pred_flow, segmask, pcd, xyz


def run_test(model, category_type, category_name, instance_type):
    result_dir = os.path.join(
        "/home/harry/discriminative_embeddings/umpmetric_results_master",
        "flownet_in_umpnet_official",
    )
    if not os.path.exists(result_dir):
        print("Creating result directory")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(os.path.join(result_dir, "succ"), exist_ok=True)
        os.makedirs(os.path.join(result_dir, "fail"), exist_ok=True)
    succ_res_dir = os.path.join(result_dir, "succ")
    fail_res_dir = os.path.join(result_dir, "fail")

    # test data info
    test_data_path = os.path.join(
        "test_data", "manipulation", category_name, instance_type
    )
    test_num = 100

    results = dict()
    sim = PybulletSim(False, 0.2)
    seq_with_correct_position = list()
    joint_type_list = list()

    for id in trange(test_num):
        animation_module = FlowNetAnimation()
        # Reset random seeds
        np.random.seed(0)

        scene_state = pickle.load(open(os.path.join(test_data_path, f"{id}.pkl"), "rb"))
        observation = sim.reset(scene_state=scene_state)
        if (
            scene_state["joint_states"][sim.selected_joint]["init_val"]
            < scene_state["joint_states"][sim.selected_joint]["cur_val"]
        ):
            print("Reversing flow prediction dir")
            reverse = True
        else:
            reverse = False
        pred_flow, segmask, pcd, xyz = predict_flow_gtmask(
            sim, observation, model, reverse
        )
        flow_norm_allpts = np.linalg.norm(pred_flow, axis=1)
        max_flow_idx = np.argpartition(
            flow_norm_allpts, -min(len(flow_norm_allpts), 10)
        )[-min(len(flow_norm_allpts), 10) :]

        if 0 in pcd[segmask].shape:
            continue

        animation_module.add_trace(
            torch.as_tensor(pcd),
            torch.as_tensor([pcd[segmask]]),
            torch.as_tensor([pred_flow] / np.linalg.norm(pred_flow[max_flow_idx])),
            "red",
        )
        max_flow_pt = (
            pcd[segmask][max_flow_idx].reshape((-1, 3)).mean(axis=0).reshape(1, 3)
        )
        action = project_pts_to_2d(
            max_flow_pt, sim.cam_view_matrix, sim._scene_cam_intrinsics
        ).reshape(
            3,
        )
        action = action[:2]
        action = np.round(action).astype("int")

        observation, (reward, move_flag), done, info = sim.step(
            [0, action[0], action[1]]
        )
        sim.save_render(sim.color_image, scene_state["instance_id"], result_dir)

        # terminate immediately if the position is wrong
        if done:
            print("fail")
            results[f"sequence-{id}"] = -1234  # specific constant for wrong position
            joint_type_list.append(None)
            save_html = animation_module.animate()
            i = scene_state["instance_id"]
            if save_html:
                save_html.write_html(
                    os.path.join(fail_res_dir, "{}.html".format(i + "_" + str(id)))
                )
            continue
        else:
            seq_with_correct_position.append(id)
            joint_type_list.append(sim.get_joint_type())

        # pre-preparation
        reach_boundary, reach_init = False, False
        bad_actions = list()
        dist2target = info["dist2init"]
        results[f"dist2target-{id}-{0}"] = dist2target

        # direction inference
        max_step_num = 2 * step_num_dict[category_name]
        for step in range(1, max_step_num + 1):
            ee_pos = np.array([sim.bc.getJointState(sim._suction_gripper, joint_id)[0] for joint_id in [0, 1, 2]]) + sim._mount_base_position
            print('Gripper loc: ', ee_pos)
            print(f"Step: {step}")
            if reach_init:
                results[f"dist2target-{id}-{step}"] = 0
                continue

            pred_flow, segmask, pcd, xyz = predict_flow_gtmask_after(
                sim, observation, model, reverse
            )
            ee_to_pt_dist = np.linalg.norm(pcd[segmask] - ee_pos, axis=1)

            flow_norm_allpts = np.linalg.norm(pred_flow, axis=1)
            flow_norm_allpts = np.divide(flow_norm_allpts, ee_to_pt_dist)
            # max_flow_idx = np.argpartition(flow_norm_allpts, -10)[-10:]
            max_flow_idx = np.argpartition(
                flow_norm_allpts, -min(len(flow_norm_allpts), 10)
            )[-min(len(flow_norm_allpts), 10) :]
            if 0 in pred_flow[max_flow_idx].shape:
                print("???")
                action = np.array([-1, 0, 0])
            else:
                action = (
                    pred_flow[max_flow_idx]
                    .reshape((-1, 3))
                    .mean(axis=0)
                    .reshape(
                        3,
                    )
                )
                action = action / np.linalg.norm(action)
                print(action)
                animation_module.add_trace(
                    torch.as_tensor(pcd),
                    torch.as_tensor([pcd[segmask]]),
                    torch.as_tensor(
                        [pred_flow] / np.linalg.norm(pred_flow[max_flow_idx])
                    ),
                    "red",
                )

            (
                observation,
                (reward, move_flag),
                (reach_init, reach_boundary),
                info,
            ) = sim.step([1, action[0], action[1], action[2]])
            sim.save_render(sim.color_image, scene_state["instance_id"], result_dir)

            if True:
                dist2target = info["dist2init"]
                results[f"dist2target-{id}-{step}"] = info["dist2init"]
                if reach_init:
                    results[f"sequence-{id}"] = step
                    print("DONE")
                    for s in range(step, max_step_num + 1):
                        results[f"dist2target-{id}-{s}"] = info["dist2init"]

                    break
        save_html = animation_module.animate()
        i = scene_state["instance_id"]
        print(
            "Normalized Distance: {}".format(
                results[f"dist2target-{id}-{max_step_num}"]
                / (1e-5 + results[f"dist2target-{id}-{0}"])
            )
        )
        if info["dist2init"] < 0.1:
            if os.path.isfile(os.path.join(result_dir, "{}.mp4".format(i))):
                shutil.move(
                    os.path.join(result_dir, "{}.mp4".format(i)),
                    os.path.join(succ_res_dir, "{}.mp4".format(i + "_" + str(id))),
                )
                if save_html:
                    save_html.write_html(
                        os.path.join(succ_res_dir, "{}.html".format(i + "_" + str(id)))
                    )
        else:
            if os.path.isfile(os.path.join(result_dir, "{}.mp4".format(i))):
                shutil.move(
                    os.path.join(result_dir, "{}.mp4".format(i)),
                    os.path.join(fail_res_dir, "{}.mp4".format(i + "_" + str(id))),
                )
                if save_html:
                    save_html.write_html(
                        os.path.join(fail_res_dir, "{}.html".format(i + "_" + str(id)))
                    )

    # result analysis
    final_result = 0

    if True:  # manipulationfor id in range(test_num):
        print(results)
        for id in range(test_num):
            if id in seq_with_correct_position:
                final_result += (
                    results[f"dist2target-{id}-{max_step_num}"]
                    / (1e-5 + results[f"dist2target-{id}-{0}"])
                    / test_num
                )
                res_file = open(os.path.join(result_dir, "umpnet_all.txt"), "a")
                number = results[f"dist2target-{id}-{max_step_num}"] / (
                    1e-5 + results[f"dist2target-{id}-{0}"]
                )
                if number > 1.0:
                    number = 1.0
                print("{}: {}".format(category_name, number), file=res_file)
                res_file.close()
            else:
                final_result += 1.0 / test_num
                res_file = open(os.path.join(result_dir, "umpnet_all.txt"), "a")
                print("{}: {}".format(category_name, 1.0, file=res_file))
                res_file.close()

    print(f"Manipulation results - {category_name}-{instance_type}: {final_result}")


if __name__ == "__main__":
    main()
