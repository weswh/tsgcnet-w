import os
import numpy as np
from plyfile import PlyData, PlyElement
import pandas as pd


def rotate_and_translate(points, normals, rot_axis, rot_angle, trans):
    # 随机旋转和平移点云
    if rot_axis == 'x':
        rot_mat = np.array([[1, 0, 0],
                            [0, np.cos(rot_angle), -np.sin(rot_angle)],
                            [0, np.sin(rot_angle), np.cos(rot_angle)]])
    elif rot_axis == 'y':
        rot_mat = np.array([[np.cos(rot_angle), 0, np.sin(rot_angle)],
                            [0, 1, 0],
                            [-np.sin(rot_angle), 0, np.cos(rot_angle)]])
    elif rot_axis == 'z':
        rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                            [np.sin(rot_angle), np.cos(rot_angle), 0],
                            [0, 0, 1]])
    else:
        raise ValueError("Invalid rotation axis, please choose from 'x', 'y' or 'z'.")
    points_rot = np.dot(points, rot_mat)
    normals_rot = np.dot(normals, rot_mat)
    points_trans = points_rot + trans
    return points_trans, normals_rot


def data_augmentation(path):
    # 定义旋转角度和平移量范围
    rot_angle_range = [-np.pi/3, np.pi/3]
    trans_range = [-10, 10]
    # 遍历路径下所有ply文件
    for filename in os.listdir(path):
        if filename.endswith(".ply"):
            # 读取ply文件
            filepath = os.path.join(path, filename)
            row_data = PlyData.read(filepath)
            points = np.array(pd.DataFrame(row_data.elements[0].data))
            xyz = points[:, :3]  # 只保留前三维坐标信息
            normals = points[:, 3:]  # 提取法向量信息
            faces = np.array(pd.DataFrame(row_data.elements[1].data))  # 面片索引加颜色
            # 进行数据增强，生成60个新的点云
            for i in range(1):
                rot_axis = np.random.choice(['x', 'y', 'z'])
                rot_angle = np.random.uniform(rot_angle_range[0], rot_angle_range[1])
                trans = np.random.uniform(trans_range[0], trans_range[1], size=(1, 3))
                points_trans, normals_rot = rotate_and_translate(xyz, normals, rot_axis, rot_angle, trans)
                # 构建新的ply文件
                new_filename = filename.split(".")[0] + "_{}.ply".format(i)
                new_filepath = os.path.join(path, new_filename)
                with open(new_filepath, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write("comment VCGLIB generated\n")
                    f.write("element vertex {}\n".format(points_trans.shape[0]))
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("property float nx\n")
                    f.write("property float ny\n")
                    f.write("property float nz\n")
                    f.write("element face {}\n".format(row_data.elements[1].count))
                    f.write("property list uchar int vertex_indices\n")
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                    f.write("property uchar alpha\n")
                    f.write("end_header\n")
                    for j in range(points_trans.shape[0]):
                        x, y, z = points_trans[j]
                        nx, ny, nz = normals_rot[j]
                        f.write("{:0.5f} {:0.5f} {:0.5f} {:0.5f} {:0.5f} {:0.5f}\n".format(x, y, z, nx, ny, nz))
                    for face in faces:
                        vertex_indices = face[0]
                        color = face[1:]
                        f.write("3 {} {} {} {} {} {} 255\n".format(vertex_indices[0], vertex_indices[1], vertex_indices[2], color[0], color[1], color[2]))
                    f.close()
            print("Augmented file {} has been generated".format(filename))



if __name__ == '__main__':
    data_path = 'data/queya'
    data_augmentation(data_path)
