import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('./')
from src.gen_rand_smples_YY import configInit, buildModel
from src import viz_utils, misc_utils, posa_utils, data_utils
from contact_optim.utils.fitting import fitting
from contact_optim.utils.optimizers import optim_factory
from utils.obj_utils import MeshData,write_obj

def checkContact(pklPath,args,args_dict,model,down_sample_fn,down_sample_fn2,up_sample_fn,up_sample_fn2,contact_z,savePath):
    device = torch.device("cuda" if args_dict.get('use_cuda') else "cpu")
    dtype = torch.float32

    ## 预先准备优化数据
    vertices, vertices_can, faces_arr, vsOld = data_utils.pkl_to_canonical(pklPath, device, dtype, **args_dict)
    vertices_can_ds = down_sample_fn.forward(vertices_can.unsqueeze(0).permute(0, 2, 1))
    vertices_can_ds = down_sample_fn2.forward(vertices_can_ds).permute(0, 2, 1).squeeze()
    
    z = torch.tensor(contact_z.astype(np.float32)).to(device)
    
    gen_batch = model.decoder(z, vertices_can_ds.expand(1, -1, -1))

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
    write_obj(savePath, meshData)

if __name__ == '__main__':
    args, args_dict = configInit()
    model, down_sample_fn, up_sample_fn, down_sample_fn2, up_sample_fn2 = buildModel(args, args_dict)

    contact_zs = np.loadtxt('./contact_optim/log.txt')
    import os
    for idx,z in enumerate(contact_zs):
        checkContact(
            R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_03\results\s007_frame_00001__00.00.00.009\smplx\000_vcl.pkl',
            args,args_dict,
            model,down_sample_fn,down_sample_fn2,up_sample_fn,up_sample_fn2,
            z[None,:],
            os.path.join(R'./contact_optim/data/4',str(idx).zfill(3)+'.obj'))