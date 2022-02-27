# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
from nbformat import write
import numpy as np
import open3d as o3d
from scipy.fftpack import cs_diff
import torch
import trimesh
import glob
import sys
sys.path.append('./')
from src import viz_utils, misc_utils, posa_utils, data_utils
from utils.obj_utils import MeshData, write_obj
from src.cmd_parser import parse_config
os.environ['POSA_dir'] = R'H:\YangYuan\Code\phy_program\POSA\POSA_dir'

def configInit():
    args, args_dict = parse_config()
    args_dict['batch_size'] = 1
    args_dict['ds_us_dir'] = os.path.expandvars(args_dict.get('ds_us_dir'))
    args_dict['rand_samples_dir'] = os.path.expandvars(args_dict.get('rand_samples_dir'))
    args_dict['model_folder'] = os.path.expandvars(args_dict.get('model_folder'))
    
    args_dict['base_dir'] = os.path.expandvars(args_dict.get('base_dir'))
    args_dict['data_dir'] = os.path.expandvars(args_dict.get('data_dir'))
    args_dict['PROX_dir'] = os.path.expandvars(args_dict.get('PROX_dir'))
    args_dict['output_dir'] = os.path.expandvars(args_dict.get('output_dir'))
    args_dict['affordance_dir'] = os.path.expandvars(args_dict.get('affordance_dir'))
    args_dict['rp_base_dir'] = os.path.expandvars(args_dict.get('rp_base_dir'))
    args_dict['checkpoint_path'] = os.path.expandvars(args_dict.get('checkpoint_path'))
    args_dict.pop('pkl_file_path')
    return args, args_dict

def buildModel(args, args_dict):
    ds_us_dir = args_dict.get('ds_us_dir')

    device = torch.device("cuda" if args_dict.get('use_cuda') else "cpu")
    dtype = torch.float32

    A_1, U_1, D_1 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 1, args_dict['use_cuda'])
    down_sample_fn = posa_utils.ds_us(D_1).to(device)
    up_sample_fn = posa_utils.ds_us(U_1).to(device)

    A_2, U_2, D_2 = posa_utils.get_graph_params(args_dict.get('ds_us_dir'), 2, args_dict['use_cuda'])
    down_sample_fn2 = posa_utils.ds_us(D_2).to(device)
    up_sample_fn2 = posa_utils.ds_us(U_2).to(device)

    faces_arr = trimesh.load(os.path.join(ds_us_dir, 'mesh_{}.obj'.format(0)), process=False).faces

    model = misc_utils.load_model_checkpoint(device=device, **args_dict).to(device)
    
    return model, down_sample_fn, up_sample_fn, down_sample_fn2, up_sample_fn2

def forward(pklPath,args,args_dict,model,down_sample_fn,down_sample_fn2,up_sample_fn,up_sample_fn2):
    pkl_file_basename = os.path.splitext(os.path.basename(pklPath))[0]
    device = torch.device("cuda" if args_dict.get('use_cuda') else "cpu")
    dtype = torch.float32
    # load pkl file
    vertices, vertices_can, faces_arr, vsOld = data_utils.pkl_to_canonical(pklPath, device, dtype, **args_dict)

    vertices_can_ds = down_sample_fn.forward(vertices_can.unsqueeze(0).permute(0, 2, 1))
    vertices_can_ds = down_sample_fn2.forward(vertices_can_ds).permute(0, 2, 1).squeeze()
    ## 隐空间变量z，待优化变量
    z = torch.tensor(np.random.normal(0, 1, (args.num_rand_samples, args.z_dim)).astype(np.float32)).to(device)
    gen_batch = model.decoder(z, vertices_can_ds.expand(args.num_rand_samples, -1, -1))

    gen_batch = gen_batch.transpose(1, 2)
    gen_batch = up_sample_fn2.forward(gen_batch)
    gen_batch = up_sample_fn.forward(gen_batch)
    gen_batch = gen_batch.transpose(1, 2)

    ## 重写--测试可视化代码，将结果保存为obj
    ## 重写--后续优化接口代码，处理成vs*1的形式？便于后续投影
    in_batch = gen_batch[0].squeeze(0)
    x = in_batch[:, 0]
    x = (x>0.5).cpu()
    meshData = MeshData()
    meshData.vert = vsOld
    meshData.face = faces_arr+1
    vertex_colors = np.ones((vertices.shape[0], 3)) * np.array([1.00, 0.75, 0.80])
    vertex_colors[x == 1, :3] = [0.0, 0.0, 1.0]
    meshData.color = vertex_colors
    write_obj('POSA.obj', meshData)

    # if args.viz:
    #     results = []
    #     for i in range(args.num_rand_samples):
    #         gen = viz_utils.show_sample(vertices_can, gen_batch[i], faces_arr, **args_dict)
    #         for m in gen:
    #             trans = np.eye(4)
    #             trans[1, 3] = 2 * i ## 让三个mesh在y轴方向上分离
    #             m.transform(trans)
    #             results.append(m)
    #     o3d.visualization.draw_geometries(results)
    # if args.render:
    #     gen_batch = gen_batch.detach().cpu().numpy()
    #     img = viz_utils.render_sample(gen_batch, vertices, faces_arr, **args_dict)
    #     for i in range(args.num_rand_samples):
    #         img[i].save(os.path.join(rand_samples_dir, '{}_sample_{:02d}.png'.format(pkl_file_basename, i)))

if __name__ == '__main__':
    args, args_dict = configInit()
    model, down_sample_fn, up_sample_fn, down_sample_fn2, up_sample_fn2 = buildModel(args, args_dict)
    forward(
        R'H:\YangYuan\项目资料\人物交互\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_01\results\s001_frame_00001__00.00.00.023\smplx\000_final.pkl',
        args,
        args_dict,
        model, down_sample_fn, down_sample_fn2, up_sample_fn, up_sample_fn2)