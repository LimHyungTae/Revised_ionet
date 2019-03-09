from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np
import os
from scipy import interpolate
from scipy.interpolate import spline
'''
b blue
g green
r red
c cyan 
m magenta
y yellow
k balck
w white
'''
# COLORSET = [(0,0,1), 'g', 'r', 'm', 'c', 'y'] #, 'k','w']
COLORSET = [(241/255.0, 101/255.0, 65/255.0), (2/255.0, 23/255.0, 157/255.0), (19/255.0, 128/255.0, 20/255.0), (191/255.0, 17/255.0, 46/255.0)]
SOFT_COLORSET = [(241/255.0, 187/255.0, 165/255.0), (174/255.0, 245/255.0, 231/255.0), (115/255.0, 123/255.0, 173/255.0), (232/255.0, 138/255.0, 139/255.0)]
LINE = ['--', '-.', ':', '-.', '-.']
# LABEL = ['LSTM', 'GRU', 'Bi-LSTM', 'Stacked Bi-LSTM']
LABEL = ['Particle Filter', 'MLP', 'Stacked Bi-LSTM', 'RONet(Ours)']
#marker o x + * s:square d:diamond p:five_pointed star
MARKER = ['p', 'x', '*', 's']
SMOOTHNESS = 30

DATA_INTERVAL = 5
def return_actual_position(x, y, direction):
    real_x = x*0.45
    real_y = y*0.45
    if direction == 'u':
        real_y += 0.058
    elif direction == 'd':
        real_y -= 0.058
    elif direction == 'l':
        real_x -= 0.058
    elif direction == 'r':
        real_x += 0.058

    return real_x, real_y

class Visualization:
    def __init__(self):
        self.color_set =  COLORSET
        self.line = LINE
        self.label = LABEL

    def setGT(self, raw_csv_file):
        gt_xyz = np.loadtxt(raw_csv_file, delimiter=',')
        #x_array: gt_xy[:,0]
        #y_array: gt_xy[:,1]
        self.gt_xyz = gt_xyz[:, -2:]

    def getSmoothedData_2D(self, x_data, y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        tck, u = interpolate.splprep([x_data, y_data], s=0)
        unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(unew, tck)

        smoothed_x = out[0].tolist()
        smoothed_y = out[1].tolist()

        return smoothed_x, smoothed_y

    def set_2D_plot_name(self, name):
        self._2d_plot_name = name

    def getRefinedData(self, data, interval):
        count = 0
        refined_data = []
        for datum in data:
            if count % interval == 0:
                refined_data.append(datum)
            count += 1
        return refined_data

    def draw_2D_trajectory(self, mode, *target_files_csv):
        global DATA_INTERVAL
        saved_file_name = self._2d_plot_name
        plot_title = "Trajectory"
        # plt.title(plot_title)
        gt_x = self.gt_xyz[:, 0]
        gt_y = self.gt_xyz[:, 1]

        plt.figure(figsize=(8, 6))
        plt.plot(gt_x, gt_y,'k',linestyle='-', label = 'GT')

        for i, csv in enumerate(target_files_csv):
            predicted_xy = np.loadtxt(csv, delimiter=',')
            x = []
            y = []
            predicted_x = predicted_xy[:, 0]
            predicted_y = predicted_xy[:, 1]
            for j in range(len(predicted_x)):
                if j % SMOOTHNESS == 0:
                    x.append(predicted_x[i])
                    y.append(predicted_y[i])

            # print(len(x))
            predicted_x = self.getRefinedData( predicted_x, DATA_INTERVAL)
            predicted_y = self.getRefinedData( predicted_y, DATA_INTERVAL)
            #
            # predicted_x, predicted_y = self.getSmoothedData_2D(predicted_x, predicted_y)
            
            #marker o x + * s:square d:diamond p:five_pointed star

            plt.plot(predicted_x, predicted_y, color = self.color_set[i], #marker= MARKER[i],
                            linestyle = LINE[i],label = self.label[i])

        data_list = [(-3, 0, 'u'), (3, -2, 'u'), (6, -4, 'l'), (6, 6, 'd'),
                     (0, 2, 'd'), (-6, -3, 'r'), (1, -5, 'l'), (-5, 5, 'r')]

        if mode == "3":
            selected_anchor = [data_list[0], data_list[3], data_list[6]]
        elif mode =="5":
            selected_anchor = [data_list[0], data_list[1], data_list[2], data_list[3], data_list[6]]
        elif mode =="8":
            selected_anchor = data_list
        for data in selected_anchor:
            real_x, real_y = return_actual_position(data[0], data[1], data[2])
            plt.scatter(real_x, real_y, c='r', marker='^', s=100)
            # axis_string = '(' + str(round(real_x, 2)) + ', ' + str(round(real_y, 2)) + ')'
            # plt.text(real_x + offset_x, real_y + offset_y, axis_string, fontsize=15)

            # plt.plot(x, y, color = self.color_set[i], #marker= marker,
            #                 linestyle = LINE[i],label = self.label[i])

        # plt.legend()

        plt.grid()
        # plt.legend(bbox_to_anchor=(1, 1),
        #            bbox_transform=plt.gcf().transFigure)
        plt.xlim(-3.0, 3.0)
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        plt.ylim(-2.5, 3.0)
        plt.xlabel("X Axis [m]", fontsize=15)
        plt.ylabel("Y Axis [m]", fontsize=15)
        fig = plt.gcf()
        # plt.show()
        fig.savefig(saved_file_name)
        print ("Done")


    def set_checked_file(self, csv_dir):
        self.target_csv = np.loadtxt(csv_dir, delimiter=',')

    def check_2D_trajectory(self):
        global DATA_INTERVAL
        saved_file_name = self._2d_plot_name
        plot_title = "Trajectory"
        # plt.title(plot_title)
        gt_x = self.target_csv[:, 2]
        gt_y = self.target_csv[:, 3]

        plt.figure(figsize=(8, 6))
        plt.plot(gt_x, gt_y,'k',linestyle='-', label = 'GT')


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

if __name__ == "__main__":
    viz = Visualization()
    # target_folder =["handbag", "handheld",
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
