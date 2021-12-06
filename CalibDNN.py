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

#MODE = "intensity"
MODE = "depth"

TRAINING = True
PREDICT = True
HISTORY = False

save_file_name = "calibDNN_model_20211206"
load_file_name = "calibDNN_model_20211206"
history_file_name = "calibDNN_model_20211206_history"


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

if HISTORY == False:

    limit_degree = 360 #degree

    limit_radian = tf.constant([limit_degree * math.pi / 180.0])

    fx = 721.5377
    fy = 721.5377
    cx = 6.095593e+02
    cy = 1.728540e+02

    image_width = 1242
    image_height = 375

    resized_width = 1242
    resized_height = 375

    lambda_translation = tf.Variable([1.0])
    lambda_pointcloud = tf.Variable([4.0])
    lambda_depthmap = tf.Variable([40.0])


    network_input_width = 1242
    network_input_height = 375
    depth_map_dim = 1
    camera_dim = 3

    BATCH_SIZE = 4
    EPOCHS = 25

    extrinsic_matrix = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                                 [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                                 [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                                 [0, 0, 0, 1]])

    rect_matrix = np.array([[9.999239e-01, 9.837760e-03, -7.445048e-03, 0],
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
    print("yaw : ", -e[0], "pitch : ", -e[1], "roll :", -e[2])

    transform_matrix = extrinsic_matrix

    saved_data_path = "C:/Users/xoz12/PycharmProjects/LidarInterpolation/"

    lidars = []
    lidars_path = []

    data_folder_path = 'C:/Users/xoz12/DNN/lidar_interpolation/data/'
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

    pointcloud_min_pointnum = 2000

    for idx, i in zip(range(len(lidars)), lidars):
        print("load pointcloud dataset.... step : ", idx + 1, " of ", len(lidars))
        # if idx > 0: continue
        for count, file in zip(range(len(i)), i):
            if count % 3 == 0:
                with open(lidars_path[idx] + file, 'rb') as f:
                    b = f.read()

                np_data = np.frombuffer(b, dt)
                df = pd.DataFrame(np_data)

                # depth map
                df_ = df.iloc[:, [0, 1, 2, 3]]

                pointcloud = df_.to_numpy().astype(np.float32)

                forward_pointcloud = pointcloud[pointcloud[:, 0] > 0.0]
                forward_pointcloud = forward_pointcloud[forward_pointcloud[:, 0] < 50.0]
                forward_pointcloud = forward_pointcloud[tf.abs(forward_pointcloud[:, 1]) < 50.0]

                if pointcloud_min_pointnum > len(forward_pointcloud):
                    pointcloud_min_pointnum = len(forward_pointcloud)

                pointcloud_dataset.append(forward_pointcloud)

    print("pointcloud_min_pointnum : ", pointcloud_min_pointnum)

    print("start loading data")

    camera_color_img = np.array([])

    transform_gt_data = []

    with open( saved_data_path + "camera_color_img", "rb") as file:
        camera_color_img = pickle.load(file)

    # new_camera_img = []
    # for i in range(len(pointcloud_dataset)):
    #     new_camera_img.append(camera_color_img[i])
    #
    # camera_color_img = np.array(new_camera_img)

    for i in range(camera_color_img.shape[0]):
        transform_gt_data.append(transform_matrix)

    transform_gt_data = np.array(transform_gt_data, dtype=np.float32)

    print("get camera_color_img img : ", camera_color_img.shape)
    print("get transform_gt_data : ", transform_gt_data.shape)
    print("finish loading data")

    input_camera_train, input_camera_test, transform_gt_train, transform_gt_test, \
          pointcloud_dataset_train, pointcloud_dataset_test = train_test_split(camera_color_img, transform_gt_data, pointcloud_dataset, test_size=0.30, random_state=10, shuffle=True)

    Train_DATA_SIZE = input_camera_train.shape[0]
    Test_DATA_SIZE = input_camera_test.shape[0]

    class ResnetBlock(Model):
        def __init__(self, channels: int, down_sample=False):
            super().__init__()
            self.__channels = channels
            self.__down_sample = down_sample
            self.__strides = [2, 1] if down_sample else [1, 1]

            KERNEL_SIZE = (3, 3)
            # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
            INIT_SCHEME = "he_normal"

            self.conv_1 = layers.Conv2D(self.__channels, strides=self.__strides[0],
                                 kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
            self.bn_1 = layers.BatchNormalization()
            self.conv_2 = layers.Conv2D(self.__channels, strides=self.__strides[1],
                                 kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
            self.bn_2 = layers.BatchNormalization()
            self.merge = layers.Add()

            if self.__down_sample:
                # perform down sampling using stride of 2, according to [1].
                self.res_conv = layers.Conv2D(
                    self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
                self.res_bn = layers.BatchNormalization()

        def call(self, inputs):
            res = inputs

            x = self.conv_1(inputs)
            x = self.bn_1(x)
            x = tf.nn.relu(x)
            x = self.conv_2(x)
            x = self.bn_2(x)

            if self.__down_sample:
                res = self.res_conv(res)
                res = self.res_bn(res)

            # if not perform down sample, then add a shortcut directly
            x = self.merge([x, res])
            out = tf.nn.relu(x)
            return out


    class ResNet18(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.conv_1 = layers.Conv2D(64, (7, 7), strides=2, padding="same", kernel_initializer="he_normal")
            self.init_bn = layers.BatchNormalization()
            self.pool_2 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
            self.res_1_1 = ResnetBlock(64)
            self.res_1_2 = ResnetBlock(64)
            self.res_2_1 = ResnetBlock(128, down_sample=True)
            self.res_2_2 = ResnetBlock(128)
            self.res_3_1 = ResnetBlock(256, down_sample=True)
            self.res_3_2 = ResnetBlock(256)
            self.res_4_1 = ResnetBlock(512, down_sample=True)
            self.res_4_2 = ResnetBlock(512)

        def call(self, inputs):
            out = self.conv_1(inputs)
            out = self.init_bn(out)
            out = tf.nn.relu(out)
            out = self.pool_2(out)
            for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
                out = res_block(out)

            return out

    class lidar_network(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mp2d = layers.MaxPool2D(pool_size=(5, 5), strides=1, padding="same")
            self.resnet18 = ResNet18()

        def call(self, inputs):
            out = self.mp2d(inputs)
            out = self.resnet18(out)
            return out

    class camera_network(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.resnet18 = ResNet18()

        def call(self, inputs):
            out = self.resnet18(inputs)
            return out

    class shared_network(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.conv3_256_1 = layers.Conv2D(256, (3, 3), padding="same")
            self.conv3_256_2 = layers.Conv2D(256, (3, 3), padding="same")
            self.conv3_512 = layers.Conv2D(512, (3, 3), padding="same")
            self.conv3_128 = layers.Conv2D(128, (3, 3), padding="same")
            self.conv3_64 = layers.Conv2D(64, (3, 3), padding="same")
            self.conv1_128_r = layers.Conv2D(128, (1, 1), padding="same")
            self.conv1_128_t = layers.Conv2D(128, (1, 1), padding="same")
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

        transformed_pointcloud = []

        for i in range(len(pointcloud)):
            point = np.array([pointcloud[i][0], pointcloud[i][1], pointcloud[i][2], 1.0])
            point = matrix.dot(point)
            new_point = np.array([point[0], point[1], point[2], pointcloud[i][3]])
            transformed_pointcloud.append(new_point)

        return np.array(transformed_pointcloud, dtype=np.float32)

    def makeDepthMap(pcd):
        new_depth_map = np.zeros((image_height, image_width, 1), np.float32)
        for i in range(len(pcd)):
            u = int(pcd[i][0] * fx / pcd[i][2] + cx)
            v = int(pcd[i][1] * fy / pcd[i][2] + cy)

            if 0 <= u < image_width:
                if 0 <= v < image_height:
                    if pcd[i][2] > 0:
                        new_depth_map[v][u] = np.array([math.sqrt(pcd[i][0]**2 + pcd[i][1]**2 + pcd[i][2]**2)], np.float32)

        new_depth_map = cv2.resize(new_depth_map, dsize=(resized_width, resized_height), interpolation=cv2.INTER_AREA)
        return new_depth_map

    def makeDepthMapBatch(pointclouds):
        depth_map_batch = []
        for i in range(len(pointclouds)):
            new_depth_map = np.zeros((image_height, image_width, 1), np.float32)
            pcd = transform(pointclouds[i],  rect_matrix.dot(transform_matrix_lidar_frame))
            for k in range(len(pcd)):
                if pcd[k][2] == 0: continue
                u = int(pcd[k][0] * fx / pcd[k][2] + cx)
                v = int(pcd[k][1] * fy / pcd[k][2] + cy)

                if 0 <= u < image_width:
                    if 0 <= v < image_height:
                        if pcd[k][2] > 0:
                            if MODE == "depth":
                                new_depth_map[v][u] = np.array([math.sqrt(pcd[k][0] ** 2 + pcd[k][1] ** 2 + pcd[k][2] ** 2)], np.float32)

                            if MODE == "intensity":
                                new_depth_map[v][u] = np.array([pcd[k][3]], np.float32)

            new_depth_map = cv2.resize(new_depth_map, dsize=(resized_width, resized_height), interpolation=cv2.INTER_AREA)
            depth_map_batch.append(new_depth_map)


        return depth_map_batch

    def makeTFBatch(matrix_batch, rand_matrix):
        new_matrix_batch = []
        for i in range(len(matrix_batch)):
            m = matrix_batch[i].dot(np.linalg.inv(rand_matrix))
            r = R.from_matrix(m[:3, :3])
            e = r.as_euler('zyx', degrees=False)

            roll = -e[2]
            pitch = -e[1]
            yaw = -e[0]

            roll = np.where(roll <= np.math.pi, roll, roll - 2*np.math.pi) # if roll > pi/2, roll - pi
            roll = np.where(roll > -np.math.pi, roll, roll + 2*np.math.pi) # if roll <= -pi/2, roll + pi
            pitch = np.where(pitch <= np.math.pi, pitch, pitch - 2*np.math.pi)  # if roll > pi/2, roll - pi
            pitch = np.where(pitch > -np.math.pi, pitch, pitch + 2*np.math.pi)  # if roll <= -pi/2, roll + pi
            yaw = np.where(yaw <= np.math.pi, yaw, yaw - 2*np.math.pi)  # if roll > pi/2, roll - pi
            yaw = np.where(yaw > -np.math.pi, yaw, yaw + 2*np.math.pi)  # if roll <= -pi/2, roll + pi

            tx = m[0][3]
            ty = m[1][3]
            tz = m[2][3]

            new_matrix = np.array([roll, pitch, yaw, tx, ty, tz])

            new_matrix_batch.append(new_matrix)

        return new_matrix_batch

    def makePointCloudTensor(pointclouds):

        sample_pointcloud = []
        for i in range(len(pointclouds)):
            sampled_pcd = random.sample(list(pointclouds[i]), pointcloud_min_pointnum)
            sample_pointcloud.append(sampled_pcd)

        return sample_pointcloud

    def randomTransform(pointclouds, matrix):
        transformed_pcd = []
        for i in range(len(pointclouds)):
            transformed_pcd.append(transform(pointclouds[i], matrix))
        return transformed_pcd

    def myLossFunction_tensor(y_true, y_pred, pointcloud):

        pointcloud = pointcloud[:, :, :3]

        def myLossFunction(y_true, y_pred):
            # loss roll pitch yaw tx ty tz

            y_true = tf.dtypes.cast(y_true, tf.float32)
            y_pred = tf.dtypes.cast(y_pred, tf.float32)

            alpha = tf.constant([0.8])
            rotation_error = tf.reduce_mean(tf.square(y_true[:, 0:3] - y_pred[:, 0:3]))
            translation_error = tf.reduce_mean(tf.square(y_true[:, 3:6] - y_pred[:, 3:6]))
            # mse = tf.keras.losses.MeanSquaredError()
            # loss = mse(y_true, y_pred)
            loss_transform = lambda_translation * (alpha * rotation_error + translation_error)
            # loss_transform = tf.multiply(lambda_translation, loss)

            # loss pointcloud
            y_true_rotation_matrix = tfg_transformation.rotation_matrix_3d.from_euler(y_true[:, 0:3])
            y_pred_rotation_matrix = tfg_transformation.rotation_matrix_3d.from_euler(y_pred[:, 0:3])

            y_true_points = tf.transpose(tf.add(tf.matmul(y_true_rotation_matrix, tf.transpose(pointcloud, perm=[0, 2, 1])), tf.expand_dims(y_true[:, 3:6], axis=2)), perm=[0, 2, 1])
            y_pred_points = tf.transpose(tf.add(tf.matmul(y_pred_rotation_matrix, tf.transpose(pointcloud, perm=[0, 2, 1])), tf.expand_dims(y_pred[:, 3:6], axis=2)), perm=[0, 2, 1])

            chamferdist = tf.reduce_mean(tf.sqrt(tfg.chamfer_distance.evaluate(y_true_points, y_pred_points)))
            loss_pointcloud = tf.multiply(lambda_pointcloud, chamferdist)

            # loss depth map
            img_width = tf.constant([1242.0])
            img_height = tf.constant([375.0])
            img_zero = tf.constant([0.0])

            fx_ = tf.constant([721.5377])
            fy_ = tf.constant([721.5377])
            cx_ = tf.constant([609.5593])
            cy_ = tf.constant([172.8540])

            y_true_depth = tf.sqrt(tf.reduce_sum(tf.square(y_true_points), axis=2))
            y_pred_depth = tf.sqrt(tf.reduce_sum(tf.square(y_pred_points), axis=2))

            # tensor ??
            y_true_width_min = (img_zero <= y_true_points[:, :, 0] * fx_ / y_true_points[:, :, 2] + cx_)
            y_true_width_max = (y_true_points[:, :, 0] * fx_ / y_true_points[:, :, 2] + cx_ < img_width)
            y_true_width = tf.logical_and(y_true_width_min, y_true_width_max)
            y_true_height_min = (img_zero <= y_true_points[:, :, 1] * fy_ / y_true_points[:, :, 2] + cy_)
            y_true_height_max = (y_true_points[:, :, 1] * fy_ / y_true_points[:, :, 2] + cy_ < img_height)
            y_true_height = tf.logical_and(y_true_height_min, y_true_height_max)
            y_true_and = tf.logical_and(y_true_width, y_true_height)
            y_true_depth_map = tf.where(y_true_and, y_true_depth, 0.0)

            y_pred_width_min = (img_zero <= y_pred_points[:, :, 0] * fx_ / y_pred_points[:, :, 2] + cx_)
            y_pred_width_max = (y_pred_points[:, :, 0] * fx_ / y_pred_points[:, :, 2] + cx_ < img_width)
            y_pred_width = tf.logical_and(y_pred_width_min, y_pred_width_max)
            y_pred_height_min = (img_zero <= y_pred_points[:, :, 1] * fy_ / y_pred_points[:, :, 2] + cy_)
            y_pred_height_max = (y_pred_points[:, :, 1] * fy_ / y_pred_points[:, :, 2] + cy_ < img_height)
            y_pred_height = tf.logical_and(y_pred_height_min, y_pred_height_max)
            y_pred_and = tf.logical_and(y_pred_width, y_pred_height)
            y_pred_depth_map = tf.where(y_pred_and, y_pred_depth, 0.0)

            N = tf.convert_to_tensor(tf.reduce_sum(tf.cast(tf.logical_or(y_true_and, y_pred_and), tf.float32)))

            loss_depth_map = tf.multiply(lambda_depthmap, tf.reduce_mean(tf.divide(tf.square(tf.subtract(y_true_depth_map, y_pred_depth_map)), N)))

            # print("LT / LP / LD : ", loss_transform.numpy(), loss_pointcloud.numpy(), loss_depth_map.numpy())
            total_loss = tf.add_n([loss_transform, loss_pointcloud, loss_depth_map])

            return total_loss

        return myLossFunction(y_true, y_pred)

    def makeNetwork():
        lidar_input = layers.Input((network_input_height, network_input_width, depth_map_dim))
        camera_input = layers.Input((network_input_height, network_input_width, camera_dim))

        lidar_model = lidar_network()(lidar_input)
        camera_model = camera_network()(camera_input)

        combined = layers.concatenate([lidar_model, camera_model])

        output = shared_network()(combined)
        final_model = Model(inputs=[lidar_input, camera_input], outputs=output)
        final_model.summary()
        return final_model

    calibDNN = makeNetwork()

    train_step = Train_DATA_SIZE // BATCH_SIZE
    test_step = Test_DATA_SIZE // BATCH_SIZE

    start_lr = 1e-3
    end_lr = 1e-6

    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        [(EPOCHS // 6) * 1 * train_step, (EPOCHS // 6) * 2 * train_step, (EPOCHS // 6) * 3 * train_step, (EPOCHS // 6) * 4 * train_step, (EPOCHS // 6) * 5 * train_step],
        [start_lr, start_lr - (start_lr - end_lr) / 6, start_lr - (start_lr - end_lr) / 6 * 2, start_lr - (start_lr - end_lr) / 6 * 3, start_lr - (start_lr - end_lr) / 6 * 4, start_lr - (start_lr - end_lr) / 6 * 5]
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0,
                                      amsgrad=False)

    loss_history = []
    min_loss = 9999999.999

    arg_ = np.array([0, 0, 0, 0, 0, 0])
    I_matrix = makeHomogenousMatrix(arg_)

    pointcloud_test = randomTransform(pointcloud_dataset_test, I_matrix)
    pointcloud_N_smaple_test = makePointCloudTensor(pointcloud_test)
    pointcloud_N_smaple_test = tf.convert_to_tensor(pointcloud_N_smaple_test)
    x_lidar_batch_test = makeDepthMapBatch(pointcloud_test)
    x_lidar_batch_test = tf.convert_to_tensor(x_lidar_batch_test)
    x_camera_batch_test = tf.convert_to_tensor(input_camera_test)
    y_TF_batch_test = makeTFBatch(transform_gt_test, I_matrix)
    y_TF_batch_test = tf.convert_to_tensor(y_TF_batch_test)

    i = tf.constant(0)
    size = tf.constant(1000)

    def condition(rotation):
        rotation_origin = rotation
        rotation = tf.transpose(rotation)
        rotation = tf.where(rotation <= np.math.pi, rotation, rotation - 2 * np.math.pi)
        rotation = tf.where(rotation > -np.math.pi, rotation, rotation + 2 * np.math.pi)
        rotation = tf.transpose(rotation)
        return tf.math.reduce_all(tf.not_equal(rotation_origin, rotation))

    def body(rotation):
        rotation = tf.transpose(rotation)
        rotation = tf.where(rotation <= np.math.pi, rotation, rotation - 2 * np.math.pi)
        rotation = tf.where(rotation > -np.math.pi, rotation, rotation + 2 * np.math.pi)
        rotation = tf.transpose(rotation)
        print("------------------")
        print(rotation)
        print("=----------------")
        return [rotation]

    if TRAINING == True:
        for epoch in range(EPOCHS):
            print("epoch : ", epoch + 1, " of " , EPOCHS)
            start = time.time()
            training_loss = 0.0

            for idx in range(train_step):
                if (idx + 1) * BATCH_SIZE > len(pointcloud_dataset_train): continue

                random_matrix = makeRandomHomogenousMatrix()
                pointcloud_batch = pointcloud_dataset_train[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                pointcloud_batch = randomTransform(pointcloud_batch, random_matrix)
                pointcloud_N_sample_tensor = makePointCloudTensor(pointcloud_batch)
                pointcloud_N_sample_tensor = tf.convert_to_tensor(pointcloud_N_sample_tensor)

                x_lidar_batch = makeDepthMapBatch(pointcloud_batch)
                x_lidar_batch = tf.convert_to_tensor(x_lidar_batch)

                x_camera_batch = input_camera_train[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                x_camera_batch = tf.convert_to_tensor(x_camera_batch)

                y_TF_batch = transform_gt_train[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                y_TF_rand_batch = makeTFBatch(y_TF_batch, random_matrix)
                y_TF_rand_batch = tf.convert_to_tensor(y_TF_rand_batch)

                model_params = calibDNN.trainable_weights
                with tf.GradientTape() as tape:
                    tape.watch(model_params)
                    output_r, output_t = calibDNN([x_lidar_batch, x_camera_batch], training=True)
                    output = tf.concat([output_r, output_t], 1)
                    loss = myLossFunction_tensor(y_TF_rand_batch, output, pointcloud_N_sample_tensor)

                grads = tape.gradient(loss, model_params)
                optimizer.apply_gradients(zip(grads, calibDNN.trainable_weights))

                training_loss = training_loss + loss

                if idx % 1 == 0:
                    print("Training loss (for one batch) at [ epoch %d ] [ %d of %d ]: [ %.4f ]" % (epoch + 1, idx, train_step - 1, float(loss)))

            total_test_loss = 0
            for k in range(test_step):
                if (k + 1) * BATCH_SIZE > len(pointcloud_dataset_test): continue
                test_result_r, test_result_t = calibDNN([x_lidar_batch_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE], x_camera_batch_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]], training=False)
                test_result = tf.concat([test_result_r, test_result_t], 1)
                test_loss = myLossFunction_tensor(y_TF_batch_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE], test_result, pointcloud_N_smaple_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE])
                total_test_loss = total_test_loss + test_loss

            total_test_loss = total_test_loss / test_step
            training_loss = training_loss / train_step
            print("Test loss : " , total_test_loss)
            print("training loss : ", training_loss)
            loss_history.append([training_loss, total_test_loss])

            print("one epoch time : ", time.time() - start)

            print("test_min_loss : ", min_loss)
            if min_loss > float(total_test_loss):
                calibDNN.save(save_file_name)
                min_loss = float(total_test_loss)
                print("save best model!")

            with open(save_file_name + "_history", "wb") as file:
                pickle.dump(loss_history, file, protocol=4)
            
            # final 1, 1, 1
            lambda_translation = lambda_translation
            lambda_pointcloud = lambda_pointcloud - (lambda_pointcloud - 1) / EPOCHS
            lambda_depthmap = lambda_depthmap - (lambda_depthmap - 1) / EPOCHS
            print("update lambda (t, p, d) : ", lambda_translation.numpy(), lambda_pointcloud.numpy(), lambda_depthmap.numpy())

    if PREDICT == True:
        print("start load model!!")
        load_model = keras.models.load_model(load_file_name)
        print("finish load model!!")
        load_model_test_output_r, load_model_test_output_t = load_model([x_lidar_batch_test[0:BATCH_SIZE], x_camera_batch_test[0:BATCH_SIZE]], training=False)
        load_model_test = tf.concat([load_model_test_output_r, load_model_test_output_t], 1)
        load_model_test_loss = myLossFunction_tensor(y_TF_batch_test[0:BATCH_SIZE], load_model_test, pointcloud_N_smaple_test[0:BATCH_SIZE])
        print("-----------------load_model_test--------------------")
        print(load_model_test)
        print("-----------------ground_truth--------------------")
        print(y_TF_batch_test[0:BATCH_SIZE])
        print("-----------------loss-----------------------------")
        print(load_model_test_loss)

if HISTORY == True:
    result_history = []
    with open(history_file_name, "rb") as file:
        result_history = pickle.load(file)

    result_history = np.array(result_history, dtype=np.float32)
    train_history = result_history[:, 0]
    test_history = result_history[:, 1]

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(train_history, 'y', label='train loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(test_history, 'b', label='test loss')
    acc_ax.set_ylabel('loss')
    acc_ax.legend(loc='upper right')

    plt.savefig(history_file_name + ".png")

    print("======================history Done============================")
