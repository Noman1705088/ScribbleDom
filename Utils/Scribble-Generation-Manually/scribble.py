
#@title Code for Scribble Generation

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import spatial
import pickle
import gdown
import os

dir_exists = os.path.isdir('Data_for_scribbling')
if not dir_exists:
    url_data = 'https://drive.google.com/drive/folders/1E2Smtb2RkR_8yRTPpusZpQHH63cVaROl?usp=sharing'
    gdown.download_folder(url_data)

print('The data for sample 151673 is downloaded. To scribble over your image, replace the downloaded files with your corresponding files. The necessary files will be generated in Colab after running the cell which prepares the data.')

'''
#Created on Apr 3, 2016

#@author: Bill BEGUERADJ, and edited by Nuwaisir
'''

sample = '151673'

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve

coords_str = ""

labels = ['WM', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'Background']
layer_colors = {
          'WM': (123, 123, 123),
          'L1': (0, 255, 0),
          'L2': (0, 123, 255),
          'L3': (255, 0, 255),
          'L4': (0, 255, 255),
          'L5': (255, 0, 0),
          'L6': (0, 0, 255),
          'Background': (200, 200, 200),
          }
label_idx = 0
paint_brush_size = 3

# dlpfc_sample = '151510'

coords = set()

# mouse callback function
def begueradj_draw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, drawing, mode, f, coords, label_idx, paint_brush_size

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_former_x, current_former_y = former_x, former_y
        # print(current_former_x, current_former_y)
        # coords += str(current_former_x) + ',' +  str(current_former_y) + ',' + str(label_idx) + '\n'
        coords.add((current_former_x, current_former_y, label_idx))
        # f.write(str(current_former_x) + ',' +  str(current_former_y) + '\n')


    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(im, (current_former_x, current_former_y), (former_x,former_y), layer_colors[labels[label_idx]], paint_brush_size)
                current_former_x = former_x
                current_former_y = former_y
                # print(current_former_x, current_former_y)
                # coords += str(current_former_x) + ',' +  str(current_former_y)  + ',' + str(label_idx) + '\n'
                # coords_int.add((current_former_x, current_former_y, label_idx))
                at_x = current_former_x
                at_y = current_former_y
                # coords.add((at_x, at_y, label_idx))
                for i in range(at_x - paint_brush_size, at_x + paint_brush_size + 1):
                    for j in range(at_y - paint_brush_size, at_y + paint_brush_size + 1):
                        coords.add((i, j, label_idx))

    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(im, (current_former_x, current_former_y), (former_x, former_y), layer_colors[labels[label_idx]], paint_brush_size)
            current_former_x = former_x
            current_former_y = former_y
            at_x = current_former_x
            at_y = current_former_y
            # coords.add((at_x, at_y, label_idx))
            for i in range(at_x - paint_brush_size, at_x + paint_brush_size + 1):
                for j in range(at_y - paint_brush_size, at_y + paint_brush_size + 1):
                    coords.add((i, j, label_idx))
    elif event == cv2.EVENT_RBUTTONDOWN:
        label_idx += 1
   
    return former_x, former_y

# human_dlpfc_folder = data_folder + 'Human_DLPFC/'

im = cv2.imread(f'Data_for_scribbling/{sample}/Histology_images/histology_image_for_scribbling_lowres.png')
# print(im)
cv2.namedWindow("Nuwaisir OpenCV")
cv2.setMouseCallback('Nuwaisir OpenCV', begueradj_draw)

while(1):
    cv2.imshow('Nuwaisir OpenCV', im)
    k = cv2.waitKey(1)&0xFF
    if k == 99:         ## press 'c' to save scribbles
        f = open(f'Data_for_scribbling/{sample}/scribble_coordinates_histology_lowres.csv', 'w')
        f.write('imagecol,imagerow,cluster.init\n')
        for i in coords:
            coords_str += str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]) + '\n'
        f.write(coords_str)
        f.close()
        break
    if k == 101:        ## press 'e' to exit scribbling
        paint_brush_size += 1
    if k == 100:        ## press 'd' to decrease brush size
        paint_brush_size -= 1
cv2.destroyAllWindows()

coordinate_file = f'Data_for_scribbling/{sample}/coordinates.csv'

coords_df = pd.read_csv(coordinate_file)
scribble_df = pd.read_csv(f'Data_for_scribbling/{sample}/scribble_coordinates_histology_lowres.csv')

def find_nearest_point(point_list, target_point):
    idx = spatial.KDTree(point_list).query(target_point)[1]
    return idx, point_list[spatial.KDTree(point_list).query(target_point)[1]]

row_span_scribble = im.shape[0]
col_span_scribble = im.shape[1]

row_span_st = coords_df['imagerow'].max() - coords_df['imagerow'].min()
col_span_st = coords_df['imagecol'].max() - coords_df['imagecol'].min()

coords_df['cluster.init'] = -1

for i in range(len(scribble_df)):
    spot = scribble_df.iloc[i]
    
    rx = spot['imagerow'] / row_span_scribble
    ry = spot['imagecol'] / col_span_scribble

    apprx_spot_x = rx * row_span_st + min(coords_df['imagerow'])
    apprx_spot_y = ry * col_span_st + min(coords_df['imagecol'])
    target_spot_idx, target_spot_coord = find_nearest_point(coords_df[['imagerow', 'imagecol']].values, (apprx_spot_x, apprx_spot_y))
    coords_df.at[target_spot_idx, 'cluster.init'] = spot['cluster.init']

# manual_scribble = coords_df[coords_df['cluster.init']]
# manual_scribble.to_csv(f'Data_for_scribbling/{sample}/manual_scribble.csv')
coords_df.to_csv(f'Data_for_scribbling/{sample}/scribble_coordinates.csv')

manual_scribble_np = np.full((max(coords_df['imagerow']) + 1, max(coords_df['imagecol']) + 1), 255)
for i in range(len(coords_df)):
    spot = coords_df.iloc[i]
    manual_scribble_np[spot['imagerow'], spot['imagecol']] = 255 if spot['cluster.init'] == -1 else spot['cluster.init']

plt.imsave(f'Data_for_scribbling/{sample}/manual_scribble.png', manual_scribble_np, cmap='tab20')

label_color_list = list(map(lambda x:list(map(lambda r:r/255, layer_colors[labels[x]])), coords_df['cluster.init']))
plt.figure(figsize=(10, 10))
# plt.scatter(coords_df['imagecol'], 700 - coords_df['imagerow'], c=label_color_list)
plt.scatter(coords_df['imagecol'], coords_df['imagerow'], c=label_color_list)
plt.savefig(f'{sample}_scribble.png')
np.save('manual_scribble_1.npy', manual_scribble_np)

# barcodes = coords_df.iloc[:, 0]
# barcode_to_scr = coords_df[['cluster.init']].astype('float')
# barcode_to_scr = barcode_to_scr.set_index(barcodes)
# barcode_to_scr['cluster.init'].replace(-1.0, np.NaN, inplace=True)

# dlpfc_samples = ['151673']

# for dlpfc_sample in dlpfc_samples:
#     # dlpfc_sample = '151670'
#     scribble_csv_file_name = 'scribble_' + dlpfc_sample + '_test_1'

#     with open('Data_for_scribbling/' + dlpfc_sample + '/pixel_barcode_map.pickle', 'rb') as handle:
#         pixel_barcode_map = pickle.load(handle)
#         print(len(pixel_barcode_map))
#         barcode_pixel_map = dict([(value, key) for key, value in pixel_barcode_map.items()])

#     # barcode_scribble_label = pd.read_csv('C:/Home/Thesis/Git_Repo/Thesis/Data/Human_DLPFC/' + dlpfc_sample + '/' + scribble_csv_file_name + '.csv')
#     barcode_scribble_label = barcode_to_scr
#     barcode_scribble_label['cluster.init'].fillna(255, inplace=True)
#     barcode_scribble_label['cluster.init'] = barcode_scribble_label['cluster.init'].astype('int')

#     # barcode_scribble_label.rename(columns={'Unnamed: 0.1': 'barcode'}, inplace=True)

#     mx_x = max([x for (x, y) in barcode_pixel_map.values()])
#     mn_x = min([x for (x, y) in barcode_pixel_map.values()])

#     mx_y = max([y for (x, y) in barcode_pixel_map.values()])
#     mn_y = min([y for (x, y) in barcode_pixel_map.values()])

#     dim_x = mx_x + 1
#     dim_y = mx_y + 1

#     scribble_matrix = np.zeros((dim_x, dim_y)) + 255

#     for index, row in barcode_scribble_label.iterrows():
#         scribble_label = row['cluster.init']
#         if index in barcode_pixel_map:
#             coords = barcode_pixel_map[index]
#             scribble_matrix[coords[0]][coords[1]] = scribble_label

#     xs = []
#     ys = []
#     labels = []

#     label_to_color = {
#         255: 'white',
#         0: 'gray',
#         1: 'red',
#         2: 'yellow',
#         3: 'black',
#         4: 'purple',
#         5: 'blue',
#         6: 'green',
#     }

#     for i in range(len(scribble_matrix)):
#         for j in range(len(scribble_matrix[0])):
#             xs.append(i)
#             ys.append(j)
#             labels.append(label_to_color[scribble_matrix[i][j]])
#     plt.figure(figsize=(10, 10))
#     plt.scatter(xs, ys, c=labels)
#     plt.savefig(scribble_csv_file_name + '.png')

#     np.save('Data_for_scribbling/' + dlpfc_sample + '/' + scribble_csv_file_name + '.npy', scribble_matrix)