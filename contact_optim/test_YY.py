import cv2
import pickle as pkl
import numpy as np
import sys
sys.path.append('./')
from utils.smpl_utils import pkl2smpl
from utils.rotate_utils import Camera_project
from utils.obj_utils import MeshData,write_obj

def draw_bbox(img, boxes, colors=None):
    if colors == None:
        for box in boxes:
            img = cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (255,0,0), 2)
        return img
    elif colors.__len__() == 1:
        for box in boxes:
            img = cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), colors, 2)
        return img
    else:
        for box,c in zip(boxes,colors):
            img = cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), c, 2)
        return img 

def calInter(box1,box2):
    if (max(box1[0],box2[0]) > min(box1[2],box2[2])):
        return None
    elif (max(box1[1],box2[1]) > min(box1[3],box2[3])):
        return None
    else:
        ix_min = max(box1[0],box2[0])
        iy_min = max(box1[1],box2[1])
        ix_max = min(box1[2],box2[2])
        iy_max = min(box1[3],box2[3])
        return [ix_min,iy_min,ix_max,iy_max]

def generateGT(vs,camEx,camIn,box):
    contact_v = np.zeros(vs.shape[0])
    vs_2d = Camera_project(vs,camEx,camIn)
    contact_v[(vs_2d[:,0]>box[0]) & (vs_2d[:,0]<box[2]) & (vs_2d[:,1]>box[1]) & (vs_2d[:,1]<box[3])] = 1
    return contact_v

if __name__ == '__main__':
    # contact_v = np.zeros(6)
    # vs_2d = np.array([
    #     [3,1],
    #     [1,2],
    #     [1.2,2.4],
    #     [0.8,3.5],
    #     [2.5,5.6],
    #     [-1.1,3.8],
    # ])
    # box = [0,0,2,4]
    # contact_v[(vs_2d[:,0]>box[0]) & (vs_2d[:,0]<box[2]) & (vs_2d[:,1]>box[1]) & (vs_2d[:,1]<box[3])] = 1

    interBox = calInter(
            [383,632,994,844],
            [711,528,928,819]
    )
    img = draw_bbox(
        cv2.imread(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\recordings\vicon_03301_03\img\s007_frame_00001__00.00.00.009.jpg'),
        [
            [383,632,994,844],
            [711,528,928,819],
            interBox
        ],
        colors=[
            [255,0,0],
            [0,255,0],
            [0,0,255]
        ]
        )
    cv2.imwrite(R'000.jpg', img)

    vs,js,fs = pkl2smpl(
        pklPath=R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_03\results\s007_frame_00001__00.00.00.009\smplx\000_vcl.pkl',
        mode='smplx',    
        )
    with open(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\fittings\mosh\vicon_03301_03\results\s007_frame_00001__00.00.00.009\smplx\000_vcl.pkl', 'rb') as file:
        data = pkl.load(file)
    contact_v = generateGT(vs,data['person00']['cam_extrinsic'],data['person00']['cam_intrinsic'],interBox)
    np.savetxt('./contact_optim/contact_gt.txt',contact_v)
    meshData = MeshData()
    meshData.vert = vs
    meshData.face = fs
    vertex_colors = np.ones((vs.shape[0], 3)) * np.array([1.00, 0.75, 0.80])
    vertex_colors[contact_v == 1, :3] = [0.0, 0.0, 1.0]
    meshData.color = vertex_colors
    write_obj('POSA_test.obj', meshData)