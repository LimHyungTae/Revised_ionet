from pandas import DataFrame
from math import atan2, sqrt, acos, cos, sin
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np
from random import shuffle
import os
import matplotlib.pyplot as plt

class DataInspector:
    def __init__(self):
        pass


    def set_2D_plot_name(self, plot_name):
        self._2d_plot_name = plot_name

    def check_2D_trajectory(self):
        global DATA_INTERVAL
        saved_file_name = self._2d_plot_name
        plot_title = "Trajectory"
        # plt.title(plot_title)
        gt_x = self.target_csv[:, 2]
        gt_y = self.target_csv[:, 3]

        plt.figure(figsize=(8, 6))
        plt.plot(gt_x, gt_y, 'k', linestyle='-', label='GT')

        plt.grid()
        # plt.legend(bbox_to_anchor=(1, 1),
        #            bbox_transform=plt.gcf().transFigure)
        plt.xlim(min(gt_x), max(gt_x))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        plt.ylim(min(gt_y), max(gt_y))
        plt.xlabel("X Axis [m]", fontsize=15)
        plt.ylabel("Y Axis [m]", fontsize=15)
        fig = plt.gcf()
        # plt.show()
        fig.savefig(saved_file_name)

def quaternion_to_yaw(x, y, z, w):

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(t3, t4)
    return yaw


def quaternion_to_yaw_array(x_array, y_array, z_array, w_array):
    t3 = +2.0 * (w_array * z_array + x_array * y_array)
    t4 = +1.0 - 2.0 * (y_array * y_array + z_array * z_array)
    yaw = np.arctan2(t3, t4)*180/np.pi
    return yaw



if __name__ == "__main__":
    vicon_csv_dir = "pocket/data2/syn/vi1.csv"
    vicon_csv = np.loadtxt(vicon_csv_dir, delimiter=',')

    vicon_xy = vicon_csv[:, 2:4]
    vicon_q_xyzw = vicon_csv [:, 4:8]

    starting_idx = 2000
    stride = 1
    iteration = 1500
    vicon_x = []
    vicon_y = []
    yaw_list = []
    yaw_list2 = []

    q_x_list = []
    q_y_list = []
    q_z_list = []
    q_w_list = []
    for i in range(iteration):
        q_x = vicon_q_xyzw[starting_idx + stride * i, 0]
        q_y = vicon_q_xyzw[starting_idx + stride * i, 1]
        q_z = vicon_q_xyzw[starting_idx + stride * i, 2]
        q_w = vicon_q_xyzw[starting_idx + stride * i, 3]

        q_x_list.append(q_x)
        q_y_list.append(q_y)
        q_z_list.append(q_z)
        q_w_list.append(q_w)

    q_x_list = np.array(q_x_list)
    q_y_list = np.array(q_y_list)
    q_z_list = np.array(q_z_list)
    q_w_list = np.array(q_w_list)

    yaw_array = quaternion_to_yaw_array(q_x_list, q_y_list, q_z_list, q_w_list)


    for i in range(iteration):
        yaw = quaternion_to_yaw(vicon_q_xyzw[starting_idx + stride * i, 0], vicon_q_xyzw[starting_idx + stride * i, 1], vicon_q_xyzw[starting_idx + stride * i, 2], vicon_q_xyzw[starting_idx + stride * i, 3])
        print(vicon_xy[starting_idx + stride * i, 0], vicon_xy[starting_idx + stride * i, 1], yaw*180/np.pi, yaw_array[i])
        vicon_x.append(vicon_xy[starting_idx + stride * i, 0])
        vicon_y.append(vicon_xy[starting_idx + stride * i, 1])

        yaw_list.append(yaw)

    print("min max")
    print(min(yaw_list), max(yaw_list))
    print (yaw_list.index(min(yaw_list)))
    print (yaw_list.index(max(yaw_list)))
    vector_x = []
    vector_y = []

    print("Yaw: ", np.array(yaw_list[:5])*180/3.14)
    for i in range(4):
        dx = vicon_x[i+1] - vicon_x[i]
        dy = vicon_y[i+1] - vicon_y[i]

        vector_x.append(dx)
        vector_y.append(dy)


    atan_angle = np.arctan2(vector_y, vector_x)* 180 /np.pi

    print(atan_angle)
    differential = atan_angle[1:] - atan_angle[:-1]
    print(differential)

    revised_differential = []
    for differential_angle in differential:
        if differential_angle > 180:
            d_theta = -360 + differential_angle
        elif differential_angle < -180:
            d_theta = 360 + differential_angle

        else:
            d_theta = differential_angle

        revised_differential.append(d_theta)

    print(revised_differential)
    plt.figure(figsize=(8, 6))
    plt.scatter(vicon_x[0], vicon_y[0], s=1000, c='r')
    plt.plot(vicon_x, vicon_y, 'k', linestyle='-', label='GT')

    ax = plt.axes()
    for x, y, yaw in zip(vicon_x, vicon_y, yaw_list):
        # ax.arrow(x, y, x+cos(yaw), y + sin(yaw), head_width=0.05, head_length=0.001)
        ax.arrow(x, y, x+cos(yaw), y + sin(yaw))


    plt.grid()

    scale = 1
    plt.xlim(min(vicon_x)*scale, max(vicon_x)*scale)
    plt.ylim(min(vicon_y)*scale, max(vicon_y)*scale)
    # plt.legend(bbox_to_anchor=(1, 1),
    #            bbox_transform=plt.gcf().transFigure)
    # plt.xticks(np.linspace(min(vicon_x), 0.005, max(vicon_x), endpoint =True))
    # plt.xticks(np.linspace(min(vicon_y),0.005,max(vicon_y), endpoint =True))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("X Axis [m]", fontsize=15)
    plt.ylabel("Y Axis [m]", fontsize=15)
    fig = plt.gcf()
    # plt.show()
    fig.savefig("test.png")




