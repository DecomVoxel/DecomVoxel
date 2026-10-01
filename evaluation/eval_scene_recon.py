import os
import json
import argparse


def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.strip().split(': ')
            result_dict[key] = float(value)
    return result_dict


parser = argparse.ArgumentParser(
    description='Arguments to evaluate total scene reconstruction metrics.'
)
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--root_path', type=str, default='./')
parser.add_argument('--data_root', type=str, default=None,
                    help='Dataset root (default: <root_path>/datasets/<dataset_type>)')
args = parser.parse_args()

dataset_type = args.dataset_type
root_path = args.root_path
exp_root_path = os.path.join(root_path, 'exps')
dataset_dir_names = {
    'replica': 'Replica',
    'scannetpp': 'Scannetpp',
}
data_root_path = args.data_root or os.path.join(
    root_path,
    'datasets',
    dataset_dir_names.get(str(dataset_type).lower(), str(dataset_type)),
)

results = {}                # for all scenes
total_scene = 0

exp_list = os.listdir(exp_root_path)
exp_list.sort()
for exp_name in exp_list:
    if f'_{dataset_type}_' not in exp_name:
        continue

    scan_id = exp_name.split('_')[-1]
    temp_exp_path = os.path.join(exp_root_path, exp_name)
    run_exp_list = os.listdir(temp_exp_path)
    run_exp_list.sort()
    run_exp_name = run_exp_list[-1]             # use the latest exp
    exp_path = os.path.join(temp_exp_path, run_exp_name, 'plots')

    mesh_list = os.listdir(exp_path)
    whole_mesh_list = [x for x in mesh_list if 'whole' in x]
    whole_mesh_list.sort(key=lambda x: int(x.split('_')[-2]))
    mesh_name = whole_mesh_list[-1]

    input_mesh = os.path.join(exp_path, mesh_name)
    gt_mesh = os.path.join(data_root_path, 'GTmesh', f'scan{scan_id}', 'scene_mesh.ply')
    output_mesh = os.path.join(exp_path, 'fuse_mesh.ply')
    output_txt = os.path.join(exp_path, 'metrics.txt')

    if dataset_type == 'replica':
        input_scalemat = os.path.join(data_root_path, f'scan{scan_id}', 'cameras.npz')
        traj = os.path.join(data_root_path, f'scan{scan_id}', 'traj.txt')
        cmd = f'python evaluation/evaluate_single_scene.py --input_mesh {input_mesh} --gt_mesh {gt_mesh} --input_scalemat {input_scalemat} --traj {traj} --output_mesh {output_mesh} --output_txt {output_txt}'
    else:
        input_scalemat = os.path.join(data_root_path, f'scan{scan_id}', 'eval_total_cameras.npz')
        cmd = f'python evaluation/evaluate_single_scene.py --input_mesh {input_mesh} --gt_mesh {gt_mesh} --input_scalemat {input_scalemat} --output_mesh {output_mesh} --output_txt {output_txt}'

    print(cmd)
    os.system(cmd)

    metrics = read_txt_to_dict(output_txt)
    for key, value in metrics.items():
        if key not in results:
            results[key] = value
        else:
            results[key] += value
    total_scene += 1

print(f'dataset: {dataset_type}, total scene: {total_scene}')
for key, value in results.items():
    print(f'{key}: {value / total_scene}')

total_scene_results_txt_path = os.path.join(exp_root_path, f'total_scene_metrics_{dataset_type}.txt')
with open(total_scene_results_txt_path, 'w') as f:
    for k, v in results.items():
        out = f"{k}: {v / total_scene}\n"
        f.write(out)
