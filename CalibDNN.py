import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import cv2
import pickle
import os
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from keras.models import Model
import math
import tensorflow_graphics.geometry.transformation as tfg_transformation
import tensorflow_graphics.nn.loss as tfg
import time
from scipy.spatial.transform import Rotation as R
import gc
from classification_models.keras import Classifiers

MODE = "depth"

TRAINING = True
PREDICT = False
DEBUG = False
HISTORY = True

# name_BATCHSIZE_EPOCHS_MINPOINTNUM_ALPHAx10_LT_LP_LD_DECAYSTEP
save_file_name = "calibDNN_paper_20220110_4_50_4000_10_4_40_1_10"
load_file_name = "calibDNN_paper_20220110_4_50_4000_10_4_40_1_10"
history_file_name = "calibDNN_paper_20220110_4_50_4000_10_4_40_1_10_history"
#
# save_file_name = "test"
# load_file_name = "test"
# history_file_name = "test"

BATCH_SIZE = 4
pointcloud_min_pointnum = 100000
pointcloud_max_pointnum = 0

end_lambda_translation = tf.Variable([1.0])
end_lambda_pointcloud = tf.Variable([1.0])
end_lambda_depthmap = tf.Variable([1.0])
alpha = tf.constant([1.0])  # multiply rotation
beta = tf.constant([1.0])

EPOCHS = 25
decay_step = 5

def RotationX(roll):
    matrix = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])
    return matrix


def RotationY(pitch):
    matrix = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])
    return matrix


def RotationZ(yaw):
    matrix = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])
    return matrix

def makeRandomHomogenousMatrix():
    roll = random.uniform(-0.174533, 0.174533)
    pitch = random.uniform(-0.174533, 0.174533)
    yaw = random.uniform(-0.174533, 0.174533)
    x = random.uniform(-0.25, 0.25)
    y = random.uniform(-0.25, 0.25)
    z = random.uniform(-0.25, 0.25)

    arg = np.array([roll, pitch, yaw, x, y, z])
    matrix = makeHomogenousMatrix(arg)

    return matrix

def makeHomogenousMatrix(args):
    R_x = RotationX(args[0])  # roll
    R_y = RotationY(args[1])  # pitch
    R_z = RotationZ(args[2])  # yaw
    T_x = args[3]  # x
    T_y = args[4]  # y
    T_z = args[5]  # z

    R = R_z.dot(R_y.dot(R_x))

    matrix = np.zeros([4, 4])
    matrix[0:3, 0:3] = R[:, :]
    matrix[0, 3] = T_x
    matrix[1, 3] = T_y
    matrix[2, 3] = T_z
    matrix[3, 3] = 1.0

    return matrix

def transform(pointcloud, matrix):
    pointcloud_tensor = tf.convert_to_tensor(pointcloud[:, :3], dtype='float32')
    intensity_tensor = tf.convert_to_tensor(pointcloud[:, 3], dtype='float32')
    matrix_tensor = tf.convert_to_tensor(matrix, dtype='float32')
    ones_tensor = tf.expand_dims(tf.ones_like(intensity_tensor), axis=1)
    pointcloud_tensor = tf.concat([pointcloud_tensor, ones_tensor], axis=1)

    transformed_pointcloud_xyz = tf.transpose(tf.matmul(matrix_tensor, tf.transpose(pointcloud_tensor, perm=[1, 0])),
                                              perm=[1, 0])
    transformed_pointcloud = tf.concat([transformed_pointcloud_xyz[:, :3], tf.expand_dims(intensity_tensor, axis=1)], 1)

    return transformed_pointcloud.numpy()

def randomTransform(pointclouds, matrix):
    transformed_pcd = []
    for i in range(len(pointclouds)):
        transformed_pcd.append(transform(pointclouds[i], matrix))
    return transformed_pcd

def makePointCloudTensor(pointclouds):
    sample_pointcloud = []
    for i in range(len(pointclouds)):
        sampled_pcd = random.sample(list(pointclouds[i]), pointcloud_min_pointnum)
        sample_pointcloud.append(sampled_pcd)

    return sample_pointcloud

def makeDepthMapBatch(pointclouds, name):
    depth_map_batch = []

    img_zero = tf.constant([0])
    zerof = tf.constant([0.0])

    if name == "0926":
        fx_ = fx
        fy_ = fy
        cx_ = cx
        cy_ = cy
        rect_matrix_ = rect_matrix
        image_width_ = image_width
        image_height_ = image_height
        img_width = tf.constant([1242])
        img_height = tf.constant([375])

    elif name =="0930":
        fx_ = fx_0930
        fy_ = fy_0930
        cx_ = cx_0930
        cy_ = cy_0930
        rect_matrix_ = rect_matrix_0930
        image_width_ = image_width_0930
        image_height_ = image_height_0930
        img_width = tf.constant([1226])
        img_height = tf.constant([370])

    for i in range(len(pointclouds)):
        if MODE == "depth" or MODE == "intensity":
            new_depth_map = np.zeros((image_height_, image_width_, 1), np.float32)
        if MODE == "both":
            new_depth_map = np.zeros((image_height_, image_width_, 2), np.float32)
        if MODE == "xyz":
            new_depth_map = np.zeros((image_height_, image_width_, 3), np.float32)
        if MODE == "xyzi":
            new_depth_map = np.zeros((image_height_, image_width_, 4), np.float32)

        # pcd = transform(pointclouds[i],  rect_matrix_)
        pcd = pointclouds[i]
        pcd = tf.convert_to_tensor(pcd)
        pcd_list = pcd[:, :3]
        intensity_list = pcd[:,3]
        depth = tf.sqrt(tf.reduce_sum(tf.square(pcd[:,:3]), axis=1))

        z_limit = (zerof < pcd[:, 2])
        width_min = (img_zero <= tf.cast(tf.math.round(pcd[:, 0] * fx_ / pcd[:, 2] + cx_), tf.int32))
        width_max = (tf.cast(tf.math.round(pcd[:, 0] * fx_ / pcd[:, 2] + cx_), tf.int32) < img_width)
        width = tf.logical_and(width_min, width_max)
        height_min = (img_zero <= tf.cast(tf.math.round(pcd[:, 1] * fy_ / pcd[:, 2] + cy_), tf.int32))
        height_max = (tf.cast(tf.math.round(pcd[:, 1] * fy_ / pcd[:, 2] + cy_), tf.int32) < img_height)
        height = tf.logical_and(height_min, height_max)
        pcd_and = tf.logical_and(width, height, z_limit)
        depth_map = tf.where(pcd_and, depth, 0.0)
        u_list = tf.cast(tf.math.round(pcd[:, 0] * fx_ / pcd[:, 2] + cx_), tf.int32)
        v_list = tf.cast(tf.math.round(pcd[:, 1] * fy_ / pcd[:, 2] + cy_), tf.int32)

        in_u_list = tf.where(pcd_and, u_list, 0)
        in_v_list = tf.where(pcd_and, v_list, 0)

        if MODE == "depth":
            new_depth_map[in_v_list, in_u_list] = tf.transpose([depth_map / max_depth], perm=[1,0])
        if MODE == "intensity":
            new_depth_map[in_v_list, in_u_list] = tf.transpose([intensity_list], perm=[1,0])
        if MODE == "both":
            new_depth_map[in_v_list, in_u_list] = tf.transpose([depth_map / max_depth, intensity_list], perm=[1, 0])
        if MODE == "xyz":
            in_point_list = tf.where(tf.expand_dims(pcd_and,1), pcd_list, [0.0])
            new_depth_map[in_v_list, in_u_list] = in_point_list
        if MODE == "xyzi":
            in_pointi_list = tf.where(tf.expand_dims(pcd_and,1), pcd, [0.0])
            new_depth_map[in_v_list, in_u_list] = in_pointi_list

        if name =="0926":
            depth_map_batch.append(tf.convert_to_tensor(new_depth_map))

        elif name == "0930":
            paddings = tf.constant([[2, 3], [8, 8], [0, 0]])
            new_depth_map = tf.pad(new_depth_map, paddings, "CONSTANT")
            depth_map_batch.append(tf.convert_to_tensor(new_depth_map))

    return depth_map_batch

def makeTFBatch(rand_matrix):
    new_matrix_batch = []
    for i in range(BATCH_SIZE):
        m = np.linalg.inv(rand_matrix)
        r = R.from_matrix(m[:3, :3])
        e = r.as_euler('zyx', degrees=False)

        roll = -e[2]
        pitch = -e[1]
        yaw = -e[0]

        tx = m[0][3]
        ty = m[1][3]
        tz = m[2][3]

        new_matrix = np.array([roll, pitch, yaw, tx, ty, tz])
        new_matrix_batch.append(new_matrix)

    return new_matrix_batch

def myLossFunction_tensor(y_true, y_pred, gt_pointcloud, rand_pointcloud, MODE):

    gt_pointcloud = gt_pointcloud[:, :, :3]
    rand_pointcloud = rand_pointcloud[:, :, :3]

    def myLossFunction(y_true, y_pred):

        y_true = tf.dtypes.cast(y_true, tf.float32)
        y_pred = tf.dtypes.cast(y_pred, tf.float32)

        rotation_error = tf.reduce_mean(tf.square(y_true[:, 0:3] - y_pred[:, 0:3]))
        translation_error = tf.reduce_mean(tf.square(y_true[:, 3:6] - y_pred[:, 3:6]))
        loss_transform = lambda_translation * (alpha * rotation_error + beta * translation_error)

        # loss pointcloud
        y_pred_rotation_matrix = tfg_transformation.rotation_matrix_3d.from_euler(y_pred[:, 0:3])
        y_pred_points = tf.transpose(tf.add(tf.matmul(y_pred_rotation_matrix, tf.transpose(rand_pointcloud, perm=[0, 2, 1])), tf.expand_dims(y_pred[:, 3:6], axis=2)), perm=[0, 2, 1])

        loss_pointcloud = tf.Variable([0.0])
        # chamferdist = tf.reduce_mean(tf.sqrt(tfg.chamfer_distance.evaluate(gt_pointcloud, y_pred_points)))
        for i in range(BATCH_SIZE):
            for k in range(100):
                start_index = (pointcloud_min_pointnum // 100) * (k)
                end_index = (pointcloud_min_pointnum // 100) * (k+1)
                loss_pointcloud = loss_pointcloud + tf.reduce_mean(tf.sqrt(tfg.chamfer_distance.evaluate(gt_pointcloud[i][start_index:end_index], y_pred_points[i][start_index:end_index])))

        loss_pointcloud = loss_pointcloud / BATCH_SIZE
        # print(loss_pointcloud.numpy())

        # loss depth map
        img_width = tf.constant([1242])
        img_height = tf.constant([375])

        fx_ = tf.constant([721.5377])
        fy_ = tf.constant([721.5377])
        cx_ = tf.constant([609.5593])
        cy_ = tf.constant([172.8540])

        #######################################################################################
        # y_true_depth = tf.sqrt(tf.reduce_sum(tf.square(gt_pointcloud), axis=2))
        # y_pred_depth = tf.sqrt(tf.reduce_sum(tf.square(y_pred_points), axis=2))
        #
        # y_true_width_min = (img_zero <= gt_pointcloud[:, :, 0] * fx_ / gt_pointcloud[:, :, 2] + cx_)
        # y_true_width_max = (gt_pointcloud[:, :, 0] * fx_ / gt_pointcloud[:, :, 2] + cx_ < img_width)
        # y_true_width = tf.logical_and(y_true_width_min, y_true_width_max)
        # y_true_height_min = (img_zero <= gt_pointcloud[:, :, 1] * fy_ / gt_pointcloud[:, :, 2] + cy_)
        # y_true_height_max = (gt_pointcloud[:, :, 1] * fy_ / gt_pointcloud[:, :, 2] + cy_ < img_height)
        # y_true_height = tf.logical_and(y_true_height_min, y_true_height_max)
        # y_true_and = tf.logical_and(y_true_width, y_true_height)
        # y_true_depth_map = tf.where(y_true_and, y_true_depth, 0.0)
        #
        # y_pred_width_min = (img_zero <= y_pred_points[:, :, 0] * fx_ / y_pred_points[:, :, 2] + cx_)
        # y_pred_width_max = (y_pred_points[:, :, 0] * fx_ / y_pred_points[:, :, 2] + cx_ < img_width)
        # y_pred_width = tf.logical_and(y_pred_width_min, y_pred_width_max)
        # y_pred_height_min = (img_zero <= y_pred_points[:, :, 1] * fy_ / y_pred_points[:, :, 2] + cy_)
        # y_pred_height_max = (y_pred_points[:, :, 1] * fy_ / y_pred_points[:, :, 2] + cy_ < img_height)
        # y_pred_height = tf.logical_and(y_pred_height_min, y_pred_height_max)
        # y_pred_and = tf.logical_and(y_pred_width, y_pred_height)
        # y_pred_depth_map = tf.where(y_pred_and, y_pred_depth, 0.0)
        #
        # N = tf.convert_to_tensor(tf.reduce_sum(tf.cast(tf.logical_or(y_true_and, y_pred_and), tf.float32)))
        #
        # loss_depth_map = tf.multiply(lambda_depthmap, tf.reduce_mean(tf.divide(tf.square(tf.subtract(y_true_depth_map, y_pred_depth_map)), N)))
        ###########################################################################################################################################

        loss_depth_map = tf.Variable([0.0])

        mse = tf.keras.losses.MeanSquaredError()

        for idx in range(BATCH_SIZE):
            img_zero = tf.constant([0])
            zerof = tf.constant([0.0])

            # step 1 make y_true_depth_map
            # y_true_depth_map = tf.Variable(tf.zeros_like((image_height, image_width, 1), dtype=tf.float32))
            y_true_depth_map = np.zeros((image_height, image_width, 1), np.float32)
            pcd = gt_pointcloud[idx]
            depth = tf.sqrt(tf.reduce_sum(tf.square(pcd[:, :3]), axis=1))

            z_limit = (zerof < pcd[:, 2])
            width_min = (img_zero <= tf.cast(tf.math.round(pcd[:, 0] * fx_ / pcd[:, 2] + cx_), tf.int32))
            width_max = (tf.cast(tf.math.round(pcd[:, 0] * fx_ / pcd[:, 2] + cx_), tf.int32) < img_width)
            width = tf.logical_and(width_min, width_max)
            height_min = (img_zero <= tf.cast(tf.math.round(pcd[:, 1] * fy_ / pcd[:, 2] + cy_), tf.int32))
            height_max = (tf.cast(tf.math.round(pcd[:, 1] * fy_ / pcd[:, 2] + cy_), tf.int32) < img_height)
            height = tf.logical_and(height_min, height_max)
            pcd_and = tf.logical_and(width, height, z_limit)
            depth_map = tf.where(pcd_and, depth, 0.0)
            u_list = tf.cast(tf.math.round(pcd[:, 0] * fx_ / pcd[:, 2] + cx_), tf.int32)
            v_list = tf.cast(tf.math.round(pcd[:, 1] * fy_ / pcd[:, 2] + cy_), tf.int32)

            in_u_list = tf.where(pcd_and, u_list, 0)
            in_v_list = tf.where(pcd_and, v_list, 0)

            y_true_depth_map[in_v_list, in_u_list] = tf.transpose([depth_map], perm=[1, 0])

            # step 1 make y_true_depth_map
            # y_pred_depth_map = tf.Variable(tf.zeros_like((image_height, image_width, 1), dtype=tf.float32))
            y_pred_depth_map = np.zeros((image_height, image_width, 1), np.float32)
            pcd = y_pred_points[idx]
            depth = tf.sqrt(tf.reduce_sum(tf.square(pcd[:, :3]), axis=1))

            z_limit = (zerof < pcd[:, 2])
            width_min = (img_zero <= tf.cast(tf.math.round(pcd[:, 0] * fx_ / pcd[:, 2] + cx_), tf.int32))
            width_max = (tf.cast(tf.math.round(pcd[:, 0] * fx_ / pcd[:, 2] + cx_), tf.int32) < img_width)
            width = tf.logical_and(width_min, width_max)
            height_min = (img_zero <= tf.cast(tf.math.round(pcd[:, 1] * fy_ / pcd[:, 2] + cy_), tf.int32))
            height_max = (tf.cast(tf.math.round(pcd[:, 1] * fy_ / pcd[:, 2] + cy_), tf.int32) < img_height)
            height = tf.logical_and(height_min, height_max)
            pcd_and = tf.logical_and(width, height, z_limit)
            depth_map = tf.where(pcd_and, depth, 0.0)
            u_list = tf.cast(tf.math.round(pcd[:, 0] * fx_ / pcd[:, 2] + cx_), tf.int32)
            v_list = tf.cast(tf.math.round(pcd[:, 1] * fy_ / pcd[:, 2] + cy_), tf.int32)

            in_u_list = tf.where(pcd_and, u_list, 0)
            in_v_list = tf.where(pcd_and, v_list, 0)

            y_pred_depth_map[in_v_list, in_u_list] = tf.transpose([depth_map], perm=[1, 0])

            loss_depth_map = loss_depth_map + mse(y_true_depth_map, y_pred_depth_map)
        loss_depth_map = loss_depth_map / BATCH_SIZE

        if MODE == "TEST":
            three_loss_history_one_epoch.append([loss_transform.numpy(), loss_pointcloud.numpy(), loss_depth_map.numpy()])

        #print("LT / LP / LD : ", loss_transform.numpy(), loss_pointcloud.numpy(), loss_depth_map.numpy())
        total_loss = tf.add_n([loss_transform, loss_pointcloud, loss_depth_map])

        return total_loss

    return myLossFunction(y_true, y_pred)

def test_0930(model):
    final_result = []
    for k in range(kitti_0930_step):
        if (k + 1) * BATCH_SIZE > len(pointcloud_dataset_0930): continue

        random_matrix = makeRandomHomogenousMatrix()
        pointcloud_batch = pointcloud_dataset_0930[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
        pointcloud_batch = randomTransform(pointcloud_batch, random_matrix)
        x_lidar_batch = makeDepthMapBatch(pointcloud_batch, "0930")
        x_lidar_batch = tf.convert_to_tensor(x_lidar_batch)
        x_camera_batch = camera_color_img_0930[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
        x_camera_batch = tf.convert_to_tensor(x_camera_batch)
        x_camera_batch = tf.image.per_image_standardization(x_camera_batch)
        y_TF_rand_batch = makeTFBatch(random_matrix)
        y_TF_rand_batch = tf.convert_to_tensor(y_TF_rand_batch, dtype=tf.float32)

        load_model_test_output_r, load_model_test_output_t = model([x_lidar_batch, x_camera_batch],
                                                                        training=False)
        load_model_test = tf.concat([load_model_test_output_r, load_model_test_output_t], 1)

        final_result.append(tf.reduce_mean(tf.math.abs(load_model_test - y_TF_rand_batch), axis=0))

    final_result = tf.convert_to_tensor(final_result)
    final_avg_error = tf.reduce_mean(final_result, axis=0)

    return final_avg_error

if TRAINING == True or PREDICT == True:
    fx = 721.5377
    fy = 721.5377
    cx = 6.095593e+02
    cy = 1.728540e+02

    image_width = 1242
    image_height = 375

    resized_width = 1242
    resized_height = 375

    image_width_0930 = 1226
    image_height_0930 = 370

    resized_width_0930 = 1226
    resized_height_0930 = 370

    start_lambda_translation = tf.Variable([4.0])
    start_lambda_pointcloud = tf.Variable([40.0])
    start_lambda_depthmap = tf.Variable([1.0])


    lambda_translation = tf.Variable([4.0])
    lambda_pointcloud = tf.Variable([40.0])
    lambda_depthmap = tf.Variable([1.0])



    pointcloud_min_x = 0.0
    pointcloud_max_x = 100.0
    pointcloud_limit_y = 100.0
    pointcloud_limit_z = 5.0

    max_depth = math.sqrt(pointcloud_max_x**2 + pointcloud_limit_y**2 + pointcloud_limit_z**2)
    min_depth = 0.0

    network_input_width = 1242
    network_input_height = 375

    if MODE == "depth" or MODE == "intensity":
        depth_map_dim = 1
    if MODE == "both":
        depth_map_dim = 2
    if MODE == "xyz":
        depth_map_dim = 3
    if MODE == "xyzi":
        depth_map_dim = 4

    camera_dim = 3

    extrinsic_matrix = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                                 [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                                 [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                                 [0, 0, 0, 1]])

    rect_matrix = np.array([[9.999239e-01, 9.837760e-03, -7.445048e-03, 0],
                            [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0],
                            [7.402527e-03, 4.351614e-03, 9.999631e-01, 0],
                            [0, 0, 0, 1]])

    Identity = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
    # kitti 0930 predict
    fx_0930 = 7.070912e+02
    fy_0930 = 7.070912e+02
    cx_0930 = 6.018873e+02
    cy_0930 = 1.831104e+02

    extrinsic_matrix_0930 = np.array([[7.027555e-03, -9.999753e-01, 2.599616e-05, -7.137748e-03],
                                      [-2.254837e-03, -4.184312e-05, -9.999975e-01, -7.482656e-02],
                                      [9.999728e-01, 7.027479e-03, -2.255075e-03, -3.336324e-01],
                                      [0, 0, 0, 1]])

    rect_matrix_0930 = np.array([[9.999239e-01, 9.837760e-03, -7.445048e-03, 0],
                                 [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0],
                                 [7.402527e-03, 4.351614e-03, 9.999631e-01, 0],
                                 [0, 0, 0, 1]])

    transform_matrix_lidar_frame = np.array([[0.0, -1.0, 0.0, 0.0],
                                             [0.0, 0.0, -1.0, 0.0],
                                             [1.0, 0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 1.0]])

    r = R.from_matrix(extrinsic_matrix[:3, :3])
    e = r.as_euler('zyx', degrees=True)
    e_r = r.as_euler('zyx', degrees=False)
    print("kitti 0926[ yaw : ", -e[0], "pitch : ", -e[1], "roll :", -e[2], " ]")

    r = R.from_matrix(extrinsic_matrix_0930[:3, :3])
    e = r.as_euler('zyx', degrees=True)
    e_r = r.as_euler('zyx', degrees=False)
    print("kitti 0930[ yaw : ", -e[0], "pitch : ", -e[1], "roll :", -e[2], " ]")

    transform_matrix = extrinsic_matrix
    transform_matrix_0930 = extrinsic_matrix_0930

    saved_data_path = "D:/kitti_dataset/"

    lidars = []
    lidars_path = []

    data_folder_path = 'D:/kitti_dataset/data_0926/'
    path = data_folder_path

    data_folder_list = os.listdir(data_folder_path)

    for folder in data_folder_list:
        path = data_folder_path + folder + '/'
        sync_file_list = os.listdir(path)

        for dir_name in sync_file_list:
            path = data_folder_path + folder + '/' + dir_name + '/' + 'data'

            if dir_name == 'velodyne_points':
                lidar_list = os.listdir(path)
                lidar_files = [file for file in lidar_list if file.endswith('.bin')]
                lidars.append(lidar_files)
                lidars_path.append(path + '/')

    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])

    pointcloud_dataset = []
    intensity_pointcloud_dataset = []

    for idx, i in zip(range(len(lidars)), lidars):
        print("load pointcloud dataset.... step : ", idx + 1, " of ", len(lidars))
        if DEBUG == True:
            if idx > 0: continue
        for count, file in zip(range(len(i)), i):
            if count % 3 == 0:
                with open(lidars_path[idx] + file, 'rb') as f:
                    b = f.read()

                np_data = np.frombuffer(b, dt)
                df = pd.DataFrame(np_data)

                # depth map
                df_ = df.iloc[:, [0, 1, 2, 3]]

                pointcloud = df_.to_numpy().astype(np.float32)

                forward_pointcloud = pointcloud[pointcloud[:, 0] > pointcloud_min_x]
                forward_pointcloud = forward_pointcloud[forward_pointcloud[:, 0] < pointcloud_max_x]
                forward_pointcloud = forward_pointcloud[tf.abs(forward_pointcloud[:, 1]) < pointcloud_limit_y]
                forward_pointcloud = forward_pointcloud[tf.abs(forward_pointcloud[:, 2]) < pointcloud_limit_z]

                if pointcloud_min_pointnum > len(forward_pointcloud):
                    pointcloud_min_pointnum = len(forward_pointcloud)

                if pointcloud_max_pointnum < len(forward_pointcloud):
                    pointcloud_max_pointnum = len(forward_pointcloud)

                forward_pointcloud = transform(forward_pointcloud, extrinsic_matrix)
                pointcloud_dataset.append(forward_pointcloud)

    # kitti 0930 dataset
    lidars = []
    lidars_path = []

    data_folder_path = 'D:/kitti_dataset/data_0930/'
    path = data_folder_path

    data_folder_list = os.listdir(data_folder_path)

    for folder in data_folder_list:
        path = data_folder_path + folder + '/'
        sync_file_list = os.listdir(path)

        for dir_name in sync_file_list:
            path = data_folder_path + folder + '/' + dir_name + '/' + 'data'

            if dir_name == 'velodyne_points':
                lidar_list = os.listdir(path)
                lidar_files = [file for file in lidar_list if file.endswith('.bin')]
                lidars.append(lidar_files)
                lidars_path.append(path + '/')

    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])

    pointcloud_dataset_0930 = []

    for idx, i in zip(range(len(lidars)), lidars):
        print("load pointcloud dataset.... step : ", idx + 1, " of ", len(lidars))
        if DEBUG == True:
            if idx > 0: continue
        for count, file in zip(range(len(i)), i):
            if count % 3 == 0:
                with open(lidars_path[idx] + file, 'rb') as f:
                    b = f.read()

                np_data = np.frombuffer(b, dt)
                df = pd.DataFrame(np_data)

                # depth map
                df_ = df.iloc[:, [0, 1, 2, 3]]

                pointcloud = df_.to_numpy().astype(np.float32)

                forward_pointcloud = pointcloud[pointcloud[:, 0] > pointcloud_min_x]
                forward_pointcloud = forward_pointcloud[forward_pointcloud[:, 0] < pointcloud_max_x]
                forward_pointcloud = forward_pointcloud[tf.abs(forward_pointcloud[:, 1]) < pointcloud_limit_y]
                forward_pointcloud = forward_pointcloud[tf.abs(forward_pointcloud[:, 2]) < pointcloud_limit_z]

                if pointcloud_min_pointnum > len(forward_pointcloud):
                    pointcloud_min_pointnum = len(forward_pointcloud)

                if pointcloud_max_pointnum < len(forward_pointcloud):
                    pointcloud_max_pointnum = len(forward_pointcloud)

                forward_pointcloud = transform(forward_pointcloud, extrinsic_matrix_0930)
                pointcloud_dataset_0930.append(forward_pointcloud)

    print("pointcloud_min_pointnum : ", pointcloud_min_pointnum)
    print("pointcloud_max_pointnum : ", pointcloud_max_pointnum)

    print("start loading data")

    print("load camera_color_img")
    camera_color_img = np.array([])
    with open( saved_data_path + "camera_color_img", "rb") as file:
        camera_color_img = pickle.load(file)

    if DEBUG == True:
        new_camera_img = []
        for i in range(len(pointcloud_dataset)):
            new_camera_img.append(camera_color_img[i])

        camera_color_img = np.array(new_camera_img)

    print("load camera_color_img_0930")
    # kitti 0930 datset
    camera_color_img_0930 = np.array([])
    camera_color_img_0930_padding = []
    with open(saved_data_path + "camera_color_img_0930_ze", "rb") as file:
        camera_color_img_0930 = pickle.load(file)
        camera_color_img_0930 = tf.convert_to_tensor(camera_color_img_0930)
        paddings = tf.constant([[2, 3], [8, 8], [0, 0]])
        for i in range(len(camera_color_img_0930)):
            pad_img = tf.pad(camera_color_img_0930[i], paddings, "CONSTANT").numpy()
            camera_color_img_0930_padding.append(pad_img)
        camera_color_img_0930 = np.array(camera_color_img_0930_padding)

    if DEBUG == True:
        new_camera_img_0930 = []
        for i in range(len(pointcloud_dataset_0930)):
            new_camera_img_0930.append(camera_color_img_0930[i])

        camera_color_img_0930 = np.array(new_camera_img_0930)

    print("finish loading data")

    pointcloud_dataset = np.array(pointcloud_dataset)
    pointcloud_dataset_0930 = np.array(pointcloud_dataset_0930)

    print("get 0926 - pointcloud_dataset : ", pointcloud_dataset.shape)
    print("get 0926 - camera_color_img img : ", camera_color_img.shape)
    print("get 0930 - pointcloud_dataset : ", pointcloud_dataset_0930.shape)
    print("get 0930 - camera_color_img img : ", camera_color_img_0930.shape)

    class lidar_network(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mp2d = layers.MaxPool2D(pool_size=(5, 5), strides=1, padding="same")
            self.conv2d = layers.Conv2D(3, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
            self.ResNet18, self.preprocess_input = Classifiers.get('resnet18')
            self.base_resnet18 = self.ResNet18(input_shape=(network_input_height, network_input_width, 3), weights='imagenet', include_top=False)

            # self.resnet18 = ResNet18()
        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'mp2d': self.mp2d,
                'conv2d': self.conv2d,
                'ResNet18': self.ResNet18,
                'base_resnet18': self.base_resnet18
            })
            return config

        def call(self, inputs):
            out = self.mp2d(inputs)
            # out = self.resnet18(out)
            out = self.conv2d(out)
            out = self.base_resnet18(out)
            return out

    class camera_network(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ResNet18, self.preprocess_input = Classifiers.get('resnet18')
            self.base_resnet18 = self.ResNet18(input_shape=(network_input_height, network_input_width, 3), weights='imagenet', include_top=False)
            # self.resnet18 = ResNet18()

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                # 'ResNet18': self.ResNet18,
                'base_resnet18': self.base_resnet18
            })
            return config

        def call(self, inputs):
            # out = self.resnet18(inputs)
            out = self.base_resnet18(inputs)
            return out


    class shared_network(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.conv3_256_1 = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal")
            self.conv3_256_2 = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal")
            self.conv3_512 = layers.Conv2D(512, (3, 3), padding="same", kernel_initializer="he_normal")
            self.conv3_128 = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")
            self.conv3_64 = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")
            self.conv1_128_r = layers.Conv2D(128, (1, 1), padding="same", kernel_initializer="he_normal")
            self.conv1_128_t = layers.Conv2D(128, (1, 1), padding="same", kernel_initializer="he_normal")
            self.dense_1280_r = layers.Dense(1280, activation="relu")
            self.dense_1280_t = layers.Dense(1280, activation="relu")
            # self.dense_rotation = layers.Dense(3, activation="sigmoid")
            # self.dense_translation = layers.Dense(3, activation="sigmoid")
            self.dense_rotation = layers.Dense(3)
            self.dense_translation = layers.Dense(3)
            self.bn1 = layers.BatchNormalization()
            self.bn2 = layers.BatchNormalization()
            self.bn3 = layers.BatchNormalization()
            self.bn4 = layers.BatchNormalization()
            self.bn5 = layers.BatchNormalization()
            self.bn6 = layers.BatchNormalization()
            self.act1 = layers.Activation('relu')
            self.act2 = layers.Activation('relu')
            self.act3 = layers.Activation('relu')
            self.act4 = layers.Activation('relu')
            self.act5 = layers.Activation('relu')
            self.act6 = layers.Activation('relu')
            self.flat1 = layers.Flatten()
            self.flat2 = layers.Flatten()
            self.drop1 = layers.Dropout(0.5)
            self.drop2 = layers.Dropout(0.5)

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'conv3_256_1': self.conv3_256_1,
                'conv3_256_2': self.conv3_256_2,
                'conv3_512': self.conv3_512,
                'conv3_128': self.conv3_128,
                'conv3_64': self.conv3_64,
                'conv1_128_r': self.conv1_128_r,
                'conv1_128_t': self.conv1_128_t,
                'dense_1280_r': self.dense_1280_r,
                'dense_1280_t': self.dense_1280_t,
                'dense_rotation': self.dense_rotation,
                'dense_translation': self.dense_translation,
                'bn1': self.bn1,
                'bn2': self.bn2,
                'bn3': self.bn3,
                'bn4': self.bn4,
                'bn5': self.bn5,
                'bn6': self.bn6,
                'act1': self.act1,
                'act2': self.act2,
                'act3': self.act3,
                'act4': self.act4,
                'act5': self.act5,
                'act6': self.act6,
                'flat1': self.flat1,
                'flat2': self.flat2,
                'drop1': self.drop1,
                'drop2': self.drop2
            })
            return config

        def call(self, inputs):
            x1 = inputs
            x = self.conv3_256_1(x1)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.conv3_512(x)
            x = self.bn2(x)
            x = self.act2(x)
            x2 = layers.concatenate([x, x1])

            x = self.conv3_128(x2)
            x = self.bn3(x)
            x = self.act3(x)
            x = self.conv3_256_2(x)
            x = self.bn4(x)
            x = self.act4(x)
            x3 = layers.concatenate([x, x2])

            x3 = self.conv3_64(x3)

            out1 = self.conv1_128_r(x3)
            out1 = self.bn5(out1)
            out1 = self.act5(out1)
            out1 = self.flat1(out1)
            out1 = self.dense_1280_r(out1)
            out1 = self.drop1(out1)
            out_rotation = self.dense_rotation(out1)

            out2 = self.conv1_128_t(x3)
            out2 = self.bn6(out2)
            out2 = self.act6(out2)
            out2 = self.flat2(out2)
            out2 = self.dense_1280_t(out2)
            out2 = self.drop2(out2)
            out_translation = self.dense_translation(out2)

            return out_rotation, out_translation

    def makeNetwork():
        lidar_input = layers.Input((network_input_height, network_input_width, depth_map_dim))
        camera_input = layers.Input((network_input_height, network_input_width, camera_dim))

        lidar_model = lidar_network()(lidar_input)
        camera_model = camera_network()(camera_input)

        combined = layers.concatenate([lidar_model, camera_model])

        out_rotation, out_translation = shared_network()(combined)
        final_model = Model(inputs=[lidar_input, camera_input], outputs=[out_rotation, out_translation])
        return final_model

    calibDNN_model = makeNetwork()
    calibDNN_model.summary()

    input_camera_train, input_camera_test, pointcloud_dataset_train, pointcloud_dataset_test = train_test_split(camera_color_img, pointcloud_dataset, test_size=0.30, random_state=10, shuffle=True)

    del camera_color_img
    del pointcloud_dataset
    gc.collect()

    Train_DATA_SIZE = input_camera_train.shape[0]
    Test_DATA_SIZE = input_camera_test.shape[0]
    kitti_0930_DATA_SIZE = camera_color_img_0930.shape[0]

    train_step = Train_DATA_SIZE // BATCH_SIZE
    test_step = Test_DATA_SIZE // BATCH_SIZE
    kitti_0930_step = kitti_0930_DATA_SIZE // BATCH_SIZE

    start_lr = 1e-2
    end_lr = 1e-6

    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        [(EPOCHS // 6) * 1 * train_step, (EPOCHS // 6) * 2 * train_step, (EPOCHS // 6) * 3 * train_step, (EPOCHS // 6) * 4 * train_step, (EPOCHS // 6) * 5 * train_step],
        [start_lr, start_lr - (start_lr - end_lr) / 6, start_lr - (start_lr - end_lr) / 6 * 2, start_lr - (start_lr - end_lr) / 6 * 3, start_lr - (start_lr - end_lr) / 6 * 4, start_lr - (start_lr - end_lr) / 6 * 5]
    )
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps= 5 * train_step,
        decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=start_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0,amsgrad=False)

    loss_history = []
    three_loss_history = []
    final_result_0926 = []
    final_result_0930 = []
    errors = []

    min_loss = 1e+15
    decay_rate = 0.5

    def exponential_decay(optimizer_, decay_rate):
        # get the optimizer configuration dictionary
        opt_cfg = optimizer_.get_config()
        """ The opt_cfg dictionary will look like this given that if you set initial learning rate to 0.1.
            {'name': 'Adam', 'learning_rate': 0.1, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 
            'epsilon': 1e-07, 'amsgrad': False}"""

        # change the value of learning rate by multiplying decay rate with learning rate to get new learning rate
        print("===================================================learning rate update!============================================================")
        print("before :", opt_cfg['learning_rate'])
        opt_cfg['learning_rate'] = opt_cfg['learning_rate'] * decay_rate
        print("after :", opt_cfg['learning_rate'])
        print("======================================================================================================================================")
        """ the changed opt_cfg dictionary will look like this, if you have initial learning rate of 0.1 and decay rate of 0.96,
            then new opt_cfg will look like this: 
            {'name': 'Adam', 'learning_rate': 0.096, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 
            'epsilon': 1e-07, 'amsgrad': False}"""
        # now just pass this updated optimizer configuartion dictionary to from_config() method and you are done, now
        # your optimzer will use new learning rate
        optimizer_ = optimizer_.from_config(opt_cfg)
        return optimizer_

    if TRAINING == True:
        for epoch in range(EPOCHS):
            print("epoch : ", epoch + 1, " of ", EPOCHS)
            start = time.time()
            training_loss = 0.0

            if epoch % decay_step == 0 and epoch != 0:
                optimizer = exponential_decay(optimizer, decay_rate)

                # start 4, 1, 40
                # final 1, 1, 1
                lambda_translation = lambda_translation - (start_lambda_translation - end_lambda_translation) / EPOCHS * decay_step
                lambda_pointcloud = lambda_pointcloud - (start_lambda_pointcloud - end_lambda_pointcloud) / EPOCHS * decay_step
                lambda_depthmap = lambda_depthmap - (start_lambda_depthmap - end_lambda_depthmap) / EPOCHS * decay_step
                print("update lambda (t, p, d) : ", lambda_translation.numpy(), lambda_pointcloud.numpy(), lambda_depthmap.numpy())

            for idx in range(train_step):
                if (idx + 1) * BATCH_SIZE > len(pointcloud_dataset_train): continue

                random_matrix = makeRandomHomogenousMatrix()
                # random_matrix = Identity
                pointcloud_batch = pointcloud_dataset_train[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                pointcloud_batch = randomTransform(pointcloud_batch, random_matrix)
                x_lidar_batch = makeDepthMapBatch(pointcloud_batch, "0926")
                x_lidar_batch = tf.convert_to_tensor(x_lidar_batch)
                x_camera_batch = input_camera_train[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                # temp_batch = x_camera_batch
                x_camera_batch = tf.convert_to_tensor(x_camera_batch)
                x_camera_batch = tf.image.per_image_standardization(x_camera_batch)

                y_TF_rand_batch = makeTFBatch(random_matrix)
                y_TF_rand_batch = tf.convert_to_tensor(y_TF_rand_batch)

                rand_pointcloud_batch = makePointCloudTensor(pointcloud_batch)
                gt_pointcloud_batch = randomTransform(np.array(rand_pointcloud_batch), np.linalg.inv(random_matrix))
                rand_pointcloud_batch = tf.convert_to_tensor(rand_pointcloud_batch)
                gt_pointcloud_batch = tf.convert_to_tensor(gt_pointcloud_batch)

                # debug
                # img = cv2.vconcat([cv2.cvtColor((x_lidar_batch[0].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB), temp_batch[0]])
                # img = cv2.cvtColor((x_lidar_batch[0].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                # img[:, :, 0] = img[:, :, 0] * 6
                # img[:, :, 1] = img[:, :, 1] * 5
                # img[:, :, 2] = img[:, :, 2] * 4
                # add_img = cv2.bitwise_or(img, temp_batch[0])
                # img = cv2.vconcat([img, add_img])
                # cv2.imshow("img", img)
                #
                # cv2.waitKey(0)

                model_params = calibDNN_model.trainable_weights
                with tf.GradientTape() as tape:
                    tape.watch(model_params)
                    output_r, output_t = calibDNN_model([x_lidar_batch, x_camera_batch], training=True)
                    output = tf.concat([output_r, output_t], 1)
                    loss = myLossFunction_tensor(y_TF_rand_batch, output, gt_pointcloud_batch, rand_pointcloud_batch, "TRAIN")

                grads = tape.gradient(loss, model_params)
                optimizer.apply_gradients(zip(grads, calibDNN_model.trainable_weights))
                training_loss = training_loss + loss

                if idx % 1 == 0:
                    print("Training loss (for one batch) at [ epoch %d ] [ %d of %d ]: [ %.4f ]" % (
                        epoch + 1, idx, train_step - 1, float(loss)))

            total_test_loss = 0
            three_loss_history_one_epoch = []
            final_result = []
            for k in range(test_step):
                if (k + 1) * BATCH_SIZE > len(pointcloud_dataset_test): continue

                random_matrix = makeRandomHomogenousMatrix()
                pointcloud_batch = pointcloud_dataset_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                pointcloud_batch = randomTransform(pointcloud_batch, random_matrix)
                x_lidar_batch = makeDepthMapBatch(pointcloud_batch, "0926")
                x_lidar_batch = tf.convert_to_tensor(x_lidar_batch)
                x_camera_batch = input_camera_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
                x_camera_batch = tf.convert_to_tensor(x_camera_batch)
                x_camera_batch = tf.image.per_image_standardization(x_camera_batch)
                y_TF_rand_batch = makeTFBatch(random_matrix)
                y_TF_rand_batch = tf.convert_to_tensor(y_TF_rand_batch, dtype=tf.float32)

                rand_pointcloud_batch = makePointCloudTensor(pointcloud_batch)
                gt_pointcloud_batch = randomTransform(np.array(rand_pointcloud_batch), np.linalg.inv(random_matrix))
                rand_pointcloud_batch = tf.convert_to_tensor(rand_pointcloud_batch)
                gt_pointcloud_batch = tf.convert_to_tensor(gt_pointcloud_batch)

                test_result_r, test_result_t = calibDNN_model([x_lidar_batch, x_camera_batch], training=False)
                test_result = tf.concat([test_result_r, test_result_t], 1)
                test_loss = myLossFunction_tensor(y_TF_rand_batch, test_result, gt_pointcloud_batch, rand_pointcloud_batch, "TEST")
                total_test_loss = total_test_loss + test_loss
                final_result.append(tf.reduce_mean(tf.math.abs(test_result - y_TF_rand_batch), axis=0))

            three_loss_history.append(tf.reduce_mean(three_loss_history_one_epoch, axis=0))
            total_test_loss = total_test_loss / test_step
            training_loss = training_loss / train_step
            print("Test loss : ", total_test_loss.numpy())
            print("training loss : ", training_loss.numpy())
            loss_history.append([training_loss, total_test_loss])

            print("one epoch time : ", time.time() - start)

            print("test_min_loss : ", min_loss)

            if min_loss > float(total_test_loss):
                calibDNN_model.save(save_file_name)
                min_loss = float(total_test_loss)
                print("save best model!")
                final_result = tf.convert_to_tensor(final_result)
                final_avg_error = tf.reduce_mean(final_result, axis=0)
                print("0926 dataset error : ", final_avg_error)
                error = test_0930(calibDNN_model)
                print("0930 dataset error : ", error)
                errors.append([[final_avg_error], [error]])

                with open(save_file_name + "_errors", "wb") as file:
                    pickle.dump(errors, file, protocol=4)

            with open(save_file_name + "_history", "wb") as file:
                pickle.dump(loss_history, file, protocol=4)

            with open(save_file_name + "_three_history", "wb") as file:
                pickle.dump(three_loss_history, file, protocol=4)

            gc.collect()

    if PREDICT == True:
        print("start load model!!")
        load_model = keras.models.load_model(load_file_name)
        print("finish load model!!")

        print("predict kitti 0926 dataset")
        final_result = []
        for k in range(test_step):
            # print(k , " step ")
            if (k + 1) * BATCH_SIZE > len(pointcloud_dataset_test): continue

            random_matrix = makeRandomHomogenousMatrix()
            pointcloud_batch = pointcloud_dataset_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
            pointcloud_batch = randomTransform(pointcloud_batch, random_matrix)
            x_lidar_batch = makeDepthMapBatch(pointcloud_batch, "0926")
            x_lidar_batch = tf.convert_to_tensor(x_lidar_batch)
            x_camera_batch = input_camera_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
            temp_batch = x_camera_batch
            x_camera_batch = tf.convert_to_tensor(x_camera_batch)
            x_camera_batch = tf.image.per_image_standardization(x_camera_batch)
            y_TF_rand_batch = makeTFBatch(random_matrix)
            y_TF_rand_batch = tf.convert_to_tensor(y_TF_rand_batch, dtype=tf.float32)

            load_model_test_output_r, load_model_test_output_t = load_model([x_lidar_batch, x_camera_batch],
                                                                            training=False)
            load_model_test = tf.concat([load_model_test_output_r, load_model_test_output_t], 1)

            # img = cv2.cvtColor((x_lidar_batch[0].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            # img[:, :, 0] = img[:, :, 0] * 6
            # img[:, :, 1] = img[:, :, 1] * 5
            # img[:, :, 2] = img[:, :, 2] * 4
            # img = cv2.vconcat([img, temp_batch[0]])
            # print("=========================================================================")
            # print(y_TF_rand_batch)
            # print(load_model_test_output_r)
            # print(load_model_test_output_t)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

            final_result.append(tf.reduce_mean(tf.math.abs(load_model_test - y_TF_rand_batch), axis=0))

        final_result = tf.convert_to_tensor(final_result)
        final_avg_error = tf.reduce_mean(final_result, axis=0)
        print("predict result : ", final_avg_error)

        final_result = []
        for k in range(kitti_0930_step):
            # print(k, " step ")
            if (k + 1) * BATCH_SIZE > len(pointcloud_dataset_0930): continue

            random_matrix = makeRandomHomogenousMatrix()
            pointcloud_batch = pointcloud_dataset_0930[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
            pointcloud_batch = randomTransform(pointcloud_batch, random_matrix)
            x_lidar_batch = makeDepthMapBatch(pointcloud_batch, "0930")
            x_lidar_batch = tf.convert_to_tensor(x_lidar_batch)
            x_camera_batch = camera_color_img_0930[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]
            x_camera_batch = tf.convert_to_tensor(x_camera_batch)
            x_camera_batch = tf.image.per_image_standardization(x_camera_batch)
            y_TF_rand_batch = makeTFBatch(random_matrix)
            y_TF_rand_batch = tf.convert_to_tensor(y_TF_rand_batch, dtype=tf.float32)
            #
            # rand_pointcloud_batch = makePointCloudTensor(pointcloud_batch)
            # gt_pointcloud_batch = randomTransform(np.array(rand_pointcloud_batch), np.linalg.inv(random_matrix))
            # rand_pointcloud_batch = tf.convert_to_tensor(rand_pointcloud_batch)
            # gt_pointcloud_batch = tf.convert_to_tensor(gt_pointcloud_batch)

            load_model_test_output_r, load_model_test_output_t = load_model([x_lidar_batch, x_camera_batch],
                                                                            training=False)
            load_model_test = tf.concat([load_model_test_output_r, load_model_test_output_t], 1)
            # load_model_test_loss = myLossFunction_tensor(y_TF_rand_batch, load_model_test, gt_pointcloud_batch, rand_pointcloud_batch, "NONE")

            final_result.append(tf.reduce_mean(tf.math.abs(load_model_test - y_TF_rand_batch), axis=0))

        final_result = tf.convert_to_tensor(final_result)
        final_avg_error = tf.reduce_mean(final_result, axis=0)
        print("predict result : ", final_avg_error)

if HISTORY == True:
    # accuarcy history
    start_index = 0
    result_history = []
    with open(save_file_name + "_history", "rb") as file:
        result_history = pickle.load(file)

    result_history = np.array(result_history, dtype=np.float32)
    train_history = result_history[start_index:, 0]
    test_history = result_history[start_index:, 1]

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(train_history, 'y', label='train loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(test_history, 'b', label='test loss')
    acc_ax.set_ylabel('loss')
    acc_ax.legend(loc='upper right')

    plt.savefig(history_file_name + "2.png")

    # local three history
    local_history = []
    with open(save_file_name + "_three_history", "rb") as file:
        local_history = pickle.load(file)

    local_history = np.array(local_history, dtype=np.float32)
    LT_history = local_history[start_index:, 0]
    LP_history = local_history[start_index:, 1]
    LD_history = local_history[start_index:, 2]

    plt.subplot(2, 2, 1)
    plt.plot(LT_history, 'y', label='LT loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')

    plt.subplot(2, 2, 2)
    plt.plot(LP_history, 'y', label='LP loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')

    plt.subplot(2, 2, 3)
    plt.plot(LD_history, 'y', label='LD loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(history_file_name + "_three2.png")

    errors = []
    with open(save_file_name + "_errors", "rb") as file:
        errors = pickle.load(file)

    print(errors)

    print("======================history Done============================")
