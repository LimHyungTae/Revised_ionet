from pandas import DataFrame
from math import atan2, sqrt, acos, cos, sin
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np
from random import shuffle
import os
import matplotlib.pyplot as plt

class DataSynchronizer:
    def set_imu_gt_data(self, imu_csv_dir, gt_csv_dir):
        self.target_imu_csv = np.loadtxt(imu_csv_dir, delimiter=',')
        self.target_gt_csv = np.loadtxt(gt_csv_dir, delimiter=',')

        assert len(self.target_imu_csv) == len(self.target_gt_csv)

    def get_atan_angle(self, gt_dx, gt_dy):
        atan_angle = np.arctan2(gt_dy, gt_dx)
        return atan_angle

    def get_theta(self, gt_xy):
        gt_dx = gt_xy[1:, 0] - gt_xy[: -1, 0]
        gt_dy = gt_xy[1:, 1] - gt_xy[: -1, 1]

        theta = self.get_atan_angle(gt_dx, gt_dy)
        return theta

    def synchronize_target_files(self):
        # Should be revise!!!!
        imu_acc_vel = self.target_imu_csv[:, :]

        gt_xy = self.target_gt_csv[:, 2:4]
        theta = self.get_theta(gt_xy)

        self.synchronized_csv = np.concatenate((imu_acc_vel, gt_xy, theta))

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    def write_file_data(self):
        result_file = open(self.out_dir, 'w', encoding='utf-8', newline='')

        wr = csv.writer(result_file)
        for i in self.synchronized_csv:
            wr.writerow(i)

        result_file.close()

if __name__ == "__main__":
    data_synchronizer = DataSynchronizer()
    target_folder_list = os.listdir('.')

    temp = []
    for target_folder in target_folder_list:
        if not "." in target_folder:
            temp.append(target_folder)
    print(temp)

    target_folder_list = temp
    target_folder_list.remove("results")

    subdirectory_data_folder_list = []
    for target_folder in target_folder_list:
        data_folder_list = os.listdir(target_folder + "/")
        temp = []
        for data_folder in data_folder_list:
            if not "." in data_folder:
                temp.append(data_folder)

        subdirectory_data_folder_list.append(temp)

    assert len(target_folder_list) == len(subdirectory_data_folder_list)

    for target_folder, data_folder_list in zip(target_folder_list, subdirectory_data_folder_list):
        try:
            for data_folder_dir in data_folder_list:
                csv_syn_dir = os.path.join(target_folder, data_folder_dir, 'syn/')
                subdirectory_csv_list = os.listdir(csv_syn_dir)

                temp = []
                for subdirectory_csv in subdirectory_csv_list:
                    if 'imu' in subdirectory_csv:
                        continue
                    else:
                        temp.append(subdirectory_csv)
                subdirectory_csv_list = temp

                for subdirectory_csv in subdirectory_csv_list:
                    csv_dir = csv_syn_dir + "/" + subdirectory_csv
                    viz.set_checked_file(csv_dir)
                    if not os.path.isdir("results/" + target_folder):
                        os.mkdir("results/" + target_folder)
                    viz.set_2D_plot_name("results/" + target_folder + "/" + data_folder_dir + "_" + subdirectory_csv[:-4]+".png")
                    viz.check_2D_trajectory()

        except ValueError:
            print("Value Error: ", target_folder, data_folder_dir, subdirectory_csv)

