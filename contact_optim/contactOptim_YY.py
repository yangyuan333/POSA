import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('./')
from src.gen_rand_smples_YY import configInit, buildModel
from src import viz_utils, misc_utils, posa_utils, data_utils
from contact_optim.utils.fitting import fitting
from contact_optim.utils.optimizers import optim_factory

def optimContac(pklPath,args,args_dict,model,down_sample_fn,down_sample_fn2,up_sample_fn,up_sample_fn2,contact_2d,lamda=0.0):
    device = torch.device("cuda" if args_dict.get('use_cuda') else "cpu")
    dtype = torch.float32

    ## 预先准备优化数据
    vertices, vertices_can, faces_arr, vsOld = data_utils.pkl_to_canonical(pklPath, device, dtype, **args_dict)
    vertices_can_ds = down_sample_fn.forward(vertices_can.unsqueeze(0).permute(0, 2, 1))
    vertices_can_ds = down_sample_fn2.forward(vertices_can_ds).permute(0, 2, 1).squeeze()
    
    ## 固定模型参数
    for p in model.parameters():
        p.requires_grad = False

    ## 优化开始
    ## 隐空间变量z，待优化变量
    ## z = torch.tensor(np.zeros((1, args.z_dim)).astype(np.float32)).to(device)
    ## z = torch.tensor(np.random.normal(0, 1, (1, args.z_dim)).astype(np.float32)).to(device)
    z = torch.tensor(np.loadtxt('./contact_optim/z.txt')[None,:].astype(np.float32)).to(device)
    z = nn.Parameter(z, requires_grad=True)
    
    ## 构建损失函数
    loss = nn.BCEWithLogitsLoss()
    ## loss = nn.BCELoss().cuda()
    ## 构建优化器
    monitor = fitting.FittingMonitor()
    optim,create_graph = optim_factory.create_optimizer([z],optim_type='sgd',lr=10)
    optim.zero_grad()
    closure = monitor.create_contact_closure(
        optim, model, up_sample_fn, up_sample_fn2, torch.tensor(contact_2d.astype(np.float32)).to(device), z, vertices_can_ds, loss, lamda=lamda, create_graph=create_graph)
    monitor.run_fitting(
        optim,
        closure,[z],None
    )

if __name__ == '__main__':
    args, args_dict = configInit()
    model, down_sample_fn, up_sample_fn, down_sample_fn2, up_sample_fn2 = buildModel(args, args_dict)

    from utils.smpl_utils import pkl2smpl
    from utils.obj_utils import MeshData,write_obj
    contact_gt = np.loadtxt('./contact_optim/contact_gt.txt')
    vs,js,fs = pkl2smpl(
        R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_03\results\s007_frame_00001__00.00.00.009\smplx\000_vcl.pkl',
        'smplx'
    )
    meshData = MeshData()
    meshData.vert = vs
    meshData.face = fs
    vertex_colors = np.ones((vs.shape[0], 3)) * np.array([1.00, 0.75, 0.80])
    vertex_colors[contact_gt == 1, :3] = [0.0, 0.0, 1.0]
    meshData.color = vertex_colors
    write_obj('./contact_optim/contact_gt.obj', meshData)

    optimContac(
        R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_03\results\s007_frame_00001__00.00.00.009\smplx\000_vcl.pkl',
        args,args_dict,
        model,down_sample_fn,down_sample_fn2,up_sample_fn,up_sample_fn2,
        contact_gt,0.001)