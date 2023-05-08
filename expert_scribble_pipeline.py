# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import sys
import numpy as np
import pandas as pd
import torch.nn.init
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
import math
from scipy import spatial
from scipy import stats
import json
import random
from inception import Inception_block

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import json

import argparse

parser = argparse.ArgumentParser(description='ScribbleSeg expert annotation pipeline')
parser.add_argument('--params', help="The input parameters json file path", required=True)

args = parser.parse_args()

with open(args.params) as f:
   params = json.load(f)
test_folder_base_name = params['test_folder_base_name']
dataset = params['dataset']
n_pcs = params['n_pcs']
scribble = params['scribble']
expert_scribble = params['expert_scribble']
nChannel = params['nChannel']
max_iter = params['max_iter']
nConv = params['nConv']
visualize = params['visualize']
use_background_scribble = params['use_background_scribble']
added_layers = params['added_layers']
last_layer_channel_count = params['last_layer_channel_count']
hyper_sum_division = params['hyper_sum_division']
seed_options = params['seed_options']
sim_options = params['sim_options']
miu_options = params['miu_options']
niu_options = params['niu_options']
lr_options = params['lr_options']

alpha_options = params['alpha_options']

refine = params['refinement_steps']

use_cuda = torch.cuda.is_available()

if use_cuda:
    print("GPU available")
else:
    print("GPU not available")

mclust_scribble = not expert_scribble
minLabels = -1 # will be assigned to the number of different scribbles used
if scribble:
    if expert_scribble: scheme = 'Expert_scribble'
    elif mclust_scribble: scheme = 'Mclust_scribble'
    else: scheme = 'Other_scribble'
else: scheme = 'No_scribble'

intermediate_channels = n_pcs # was n_pcs

meta_data_index = ['test_name', 'seed', 'dataset', 'sample', 'n_pcs', 'scribble', 'max_iter', 'sim', 'miu', 'niu','alpha', 'scheme', 'lr', 'nConv', 'no_of_scribble_layers', 'intermediate_channels', 'added_layers', 'last_layer_channel_count', 'hyper_sum_division']

test_name = f'{test_folder_base_name}_itr_{max_iter}'
# seed_options = pd.read_csv('./Data/seed_list.csv')['seeds'].values

samples = params['samples']

models = []
for sample in samples:
    for seed in seed_options:
        for sim in sim_options:
            for miu in miu_options:
                for niu in niu_options:
                    for lr in lr_options:
                        for alpha in alpha_options:
                            models.append(
                                {
                                    'seed': seed,
                                    'stepsize_sim': sim,
                                    'stepsize_con': miu,
                                    'stepsize_scr': niu,
                                    'lr': lr,
                                    'sample': sample,
                                    'alpha': alpha
                                }
                            )

# %%
report_list = []
for model in tqdm(models):
    report_map = {}
    seed = model['seed']
    lr = model['lr']
    stepsize_sim = model['stepsize_sim']
    stepsize_con = model['stepsize_con']
    stepsize_scr = model['stepsize_scr']
    sample = model['sample']

    alpha = model['alpha']

    print("************************************************")
    print('Model description:')
    print(f'sample: {sample}')
    print(f'seed: {seed}')
    print(f'lr: {lr}')
    # print(f'sim: {stepsize_sim}')
    # print(f'miu: {stepsize_con}')
    # print(f'niu: {stepsize_scr}')
    print(f'alpha: {alpha}')

    report_map['sample'] = sample
    report_map['seed'] = seed
    report_map['sim'] = stepsize_sim
    report_map['miu'] = stepsize_con
    report_map['niu'] = stepsize_scr
    report_map['alpha'] = alpha
    report_map['refinement_steps'] = refine

    npz_path = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Npzs'
    npy_path = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Npys'
    pickle_path = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Pickles'
    coordinates_file_name = 'coordinates.csv'

    # %%
    def make_directory_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)

    # scribble_img = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Scribble/manual_scribble_mclust_10_percent.npy'
    scribble_img = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Scribble/manual_scribble_1.npy'

    if mclust_scribble:
        scribble_img = f'Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data/{dataset}/{sample}/Scribble/mclust_scribble.npy'
    local_data_folder_path = './Algorithms/Unsupervised_Segmentation/Approaches/With_Scribbles/Local_Data'

    input = f'{npy_path}/mapped_{n_pcs}.npy'
    inv_xy = f'{pickle_path}/inv_spot_xy.pickle'
    border = npz_path+'/borders.npz'
    background = npy_path+'/backgrounds.npy'
    foreground = npy_path+'/foregrounds.npy'
    indices_arg = npy_path+'/indices.npy'
    pixel_barcode_map_path = pickle_path+'/pixel_barcode_map.pickle'
    coordinate_file = f'Data/{dataset}/{sample}/{coordinates_file_name}'
    map_pixel_to_grid_spot_file_path = f'{local_data_folder_path}/{dataset}/{sample}/Jsons/map_pixel_to_grid_spot.json'
    pixel_barcode_file_path = f'{local_data_folder_path}/{dataset}/{sample}/Npys/pixel_barcode.npy'
    manual_annotation_file_path = f'./Data/{dataset}/{sample}/manual_annotations.csv'

    output_folder_path = f'./Outputs/{test_name}/{dataset}/{sample}'
    # leaf_output_folder_path = f'{output_folder_path}/{scheme}/{n_pcs}_pcs/Seed_{seed}/Lr_{lr}/Hyper_{stepsize_sim}_{stepsize_con}_{stepsize_scr}'
    leaf_output_folder_path = f'{output_folder_path}/{scheme}/{n_pcs}_pcs/Seed_{seed}/Lr_{lr}/Hyper_{alpha}'
    labels_per_itr_folder_path = f'{leaf_output_folder_path}/Labels_per_itr/'
    image_per_itr_folder_path = f'{leaf_output_folder_path}/Image_per_itr/'
    meta_data_file_path = f'{leaf_output_folder_path}/meta_data.csv'

    # %%
    pixel_barcode = np.load(pixel_barcode_file_path)
    pixel_rows_cols = np.argwhere(pixel_barcode != '')
    df_man = pd.read_csv(manual_annotation_file_path, index_col=0)
    manual_annotation_labels = df_man['label'].values
    ari_per_itr = []
    loss_per_itr = []
    df_barcode_labels_per_itr = pd.DataFrame(index = pixel_barcode[pixel_barcode != ''])
    backgrounds = np.load(background)
    foregrounds = np.load(foreground)

    # %%
    make_directory_if_not_exist(output_folder_path)
    make_directory_if_not_exist(labels_per_itr_folder_path)
    make_directory_if_not_exist(image_per_itr_folder_path)
    make_directory_if_not_exist("./Report")

    # %%
    with open(map_pixel_to_grid_spot_file_path, 'r') as f:
        map_pixel_to_grid_spot = json.load(f)

    # %%
    def make_str(x):
        return f'({x[0]}, {x[1]})'

    def get_grid_spots_from_pixels(pixels, colors):
        grid_spots = np.array([map_pixel_to_grid_spot[make_str(pixel)] for pixel in pixels if make_str(pixel) in map_pixel_to_grid_spot])
        predicted_colors = [colors[i] for i in range(len(pixels)) if make_str(pixels[i]) in map_pixel_to_grid_spot]
        return grid_spots, predicted_colors

    # %%
    def calc_ari(df_1, df_2):
        df_merged = pd.merge(df_1, df_2, left_index=True, right_index=True).dropna()
        cols = df_merged.columns
        for col in cols:
            df_merged[col] = df_merged[col].values.astype('int')
        return adjusted_rand_score(df_merged[cols[0]].values, df_merged[cols[1]].values)

    # %%
    torch.manual_seed(seed)
    np.random.seed(seed)

    no_of_scribble_layers = 0

    # CNN model
    class MyNet(nn.Module):
        def __init__(self,input_dim):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(input_dim, intermediate_channels, kernel_size=1, stride=1, padding=0 )
            self.bn1 = nn.BatchNorm2d(intermediate_channels)

            # inception_block(in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool)

            self.inception3a = Inception_block(intermediate_channels, 160, 96, 64, 16, 16, 16)
            # self.inception3a = Inception_block(intermediate_channels, 253, 96, 1, 16, 1, 1)
            # self.conv2 = nn.Conv2d(intermediate_channels, 256, kernel_size=1, stride=1, padding=0 )
            self.bn_i_1 = nn.BatchNorm2d(256)

            self.inception3b = nn.ModuleList()
            self.bn_i_2 = nn.ModuleList()

            if nConv >= 1:
                self.inception3b.append(Inception_block(256, 96, 32, 16, 16, 8, 8))
                # self.inception3b.append(Inception_block(256, 125, 32, 1, 16, 1, 1))
                self.bn_i_2.append(nn.BatchNorm2d(128))

                for i in range(nConv-1):
                    self.inception3b.append(Inception_block(128, 96, 32, 16, 16, 8, 8))
                    # self.inception3b.append(Inception_block(128, 125, 32, 1, 16, 1, 1))
                    self.bn_i_2.append(nn.BatchNorm2d(128))

            r = last_layer_channel_count

            print('last layer size:', r)
            if nConv>=1:
                self.conv3 = nn.Conv2d(128, r, kernel_size=1, stride=1, padding=0 )
            else:
                self.conv3 = nn.Conv2d(256, r, kernel_size=1, stride=1, padding=0 )
            self.bn3 = nn.BatchNorm2d(r)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu( x )
            x = self.bn1(x)
            
            x = self.inception3a(x)
            # x = self.conv2(x)
            x = F.relu( x )
            x = self.bn_i_1(x)

            for i in range(nConv):
                x = self.inception3b[i](x)
                x = F.relu( x )
                x = self.bn_i_2[i](x)

            x = self.conv3(x)
            x = self.bn3(x)
            return x

    # %%
    im = np.load(input)
    im.shape

    # %%
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')]) ) # z, y, x
    data.shape

    # %%
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    data.shape

    # %%
    def relabel_mask(mask, background_val):
        row, col = mask.shape
        mask = mask.reshape(-1)
        values = np.unique(mask[mask != background_val])
        lookup = {k: v for v, k in enumerate(dict.fromkeys(values))}
        lookup[background_val] = background_val
        mask = np.array([lookup[i] for i in mask])
        return mask.reshape(row, col)

    # %%
    # load scribble
    if scribble:
        mask = np.load(scribble_img)
        foreground_val = 1000
        background_val = 255
        mask = relabel_mask(mask.copy(), background_val)
        if len(mask[mask != background_val]) == 0:
            print('Expecting some scribbles, but no scribbles are found!')
            last_layer_channel_count = 100 + added_layers
            nChannel = last_layer_channel_count
        else:
            
            mask_foreground = mask.copy()
            mask_foreground[foregrounds[:, 0], foregrounds[:, 1]] = foreground_val
            
            mx_label_num = mask[mask != background_val].max()
            if use_background_scribble:
                mask[backgrounds[:, 0], backgrounds[:, 1]] = mx_label_num + 1 # Assuming that scribble labels increase by 1
            mask = mask.reshape(-1)
            scr_idx = np.where(mask != 255)[0]
            mask_foreground = mask_foreground.reshape(-1)

            mask_inds = np.unique(mask)
            mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==background_val) )

            for i in range(1, len(mask_inds)):
                if mask_inds[i] - mask_inds[i-1] != 1:
                    print("Problem in scribble labels. Not increasing by 1.")

            # # Take all of the foreground into similarity component
            # inds_sim = torch.from_numpy( np.where( mask_foreground == foreground_val )[ 0 ] ) # Big change done!

            # # Take the non-scribbled foreground into similarity component
            mask_foreground[scr_idx] = background_val
            inds_sim = torch.from_numpy( np.where( mask_foreground == foreground_val )[ 0 ] ) # Big change done!

            # inds_sim = torch.from_numpy( np.where( mask == background_val )[ 0 ] )
            inds_scr = torch.from_numpy( np.where( mask != background_val )[ 0 ] )
            inds_scr_array = [None for _ in range(mask_inds.shape[0])]

            for i in range(mask_inds.shape[0]):
                inds_scr_array[i] = torch.from_numpy( np.where( mask == mask_inds[i] )[ 0 ] )

            target_scr = torch.from_numpy( mask.astype(np.int64) )

            if use_cuda:
                inds_sim = inds_sim.cuda()
                inds_scr = inds_scr.cuda()
                target_scr = target_scr.cuda()


            target_scr = Variable( target_scr ) # *************** Why? **************

            minLabels = len(mask_inds)
            # nChannel = minLabels + 1
            nChannel = minLabels + added_layers # ************ Change ************ 

            no_of_scribble_layers = minLabels # **************** Addition *****************
            last_layer_channel_count = no_of_scribble_layers + added_layers
    else:
        # last_layer_channel_count = 100 + added_layers
        last_layer_channel_count = added_layers
        nChannel = last_layer_channel_count



    # %%
    # train
    model = MyNet( data.size(1) )
    if use_cuda:
        model.cuda()
    model.train()

    # %%
    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # scribble loss definition
    loss_fn_scr = torch.nn.CrossEntropyLoss()

    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(reduction='mean')
    loss_hpz = torch.nn.L1Loss(reduction='mean')
    # loss for the diagonal neighbour
    loss_hp_diag = torch.nn.L1Loss(reduction='mean')

    HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, nChannel)
    HP_diag_target = torch.zeros(im.shape[0] - 1, im.shape[1] - 1, nChannel)
    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()
        HP_diag_target = HP_diag_target.cuda()
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    label_colours = np.random.randint(255,size=(255,3))

    label_colours[0,:] = [255,255,255]
    label_colours[1,:] = [0,255,0]
    label_colours[2,:] = [255,0,0]
    label_colours[3,:] = [255,255,0]
    label_colours[4,:] = [0,255,255]
    label_colours[5,:] = [255,0,255]
    label_colours[6,:] = [0,0,0]
    label_colours[7,:] = [73,182,255]

    loss_comparison = 0

    # %%
    if sample != 'Melanoma':
        borders = np.load(border)

        right_border = borders['right_border']
        left_border = borders['left_border']
        up_border = borders['up_border']
        down_border = borders['down_border']

    if sample != 'Melanoma':
        nw_border = borders['nw_border']
        se_border = borders['se_border']

    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    loss_list = []
    loss_without_hyperparam_list = []

    for batch_idx in (range(max_iter)):

        # forwarding
        optimizer.zero_grad()   # ******************** check ********************

        output = model( data )[ 0 ]
        output[:, backgrounds[:, 0], backgrounds[:, 1]] = 0 # Big problem, as all these 1s will be normalized

        output = output.permute( 1, 2, 0 )
        output = output.contiguous().view( -1, nChannel )

        outputHP = output.reshape( (im.shape[0], im.shape[1], nChannel) )


        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        if sample != 'Melanoma':
            HPy[up_border[:, 0] - 1, up_border[:, 1], :] = 0
            HPy[down_border[:, 0], down_border[:, 1], :] = 0

        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        if sample != 'Melanoma':
            HPz[left_border[:, 0], left_border[:, 1] - 1, :] = 0
            HPz[right_border[:, 0], right_border[:, 1], :] = 0
        
        if sample != 'Melanoma':
            HP_diag = outputHP[1:,1:, :] - outputHP[0:-1, 0:-1, :]
            HP_diag[nw_border[:, 0] - 1, nw_border[:, 1] - 1, :] = 0
            HP_diag[se_border[:, 0], se_border[:, 1], :] = 0


        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)
        lhp_diag = 0
        if sample != 'Melanoma':
            lhp_diag = loss_hp_diag(HP_diag, HP_diag_target)
        

        ignore, target = torch.max( output, 1 )


        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        im_ari = im_target.reshape(im.shape[0], im.shape[1])

        
        im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])
        labels = im_cluster_num[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]
        df_labels = pd.DataFrame({'label': labels}, index=pixel_barcode[pixel_barcode != ''])
        ari_per_itr.append(calc_ari(df_man, df_labels))
        df_barcode_labels_per_itr[f'itr_{batch_idx}'] = labels

        if visualize and (batch_idx<10 or batch_idx%10 == 0):
        
            im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])
            labels = im_cluster_num[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]
            grid_spots, colors = get_grid_spots_from_pixels(pixel_rows_cols, labels)
            if dataset == 'Custom': rad = 700
            else: rad = 10
            plt.figure(figsize=(5.5,5))
            plt.scatter(grid_spots[:, 1], 1000 - grid_spots[:, 0], c=colors, s=rad)
            plt.axis('off')
            plt.savefig(f'{image_per_itr_folder_path}/itr_{batch_idx}.png',format='png',dpi=1200,bbox_inches='tight',pad_inches=0)
            plt.close('all')

        # loss 
        if scribble:        

    
            loss_lr = 0
            for i in range(mask_inds.shape[0]):
                loss_lr += loss_fn_scr(output[ inds_scr_array[i] ], target_scr[ inds_scr_array[i] ])

            loss_sim = loss_fn(output[ inds_sim ], target[ inds_sim ])
            hyper_sum = stepsize_sim + stepsize_scr + stepsize_con

            sim_multiplier = 1
            con_multiplier = 1
            scr_multiplier = 1
            L_sim = stepsize_sim * loss_sim * sim_multiplier
            L_scr = stepsize_scr * loss_lr * scr_multiplier

            L_con = stepsize_con * (lhpy + lhpz + lhp_diag) * con_multiplier
            loss_without_hyperparam = loss_sim + loss_lr + (lhpy + lhpz + lhp_diag)

            if hyper_sum_division:
                loss = (L_sim + L_con + L_scr) / hyper_sum
            else:
                loss = (L_sim + L_con + L_scr)

            # loss = alpha * loss_sim + (1 - alpha) * loss_lr + (lhpy + lhpz + lhp_diag)/3
            loss = alpha * loss_sim + (1 - alpha) * loss_lr
            # print(f'loss_sim: {loss_sim}, loss_lr: {loss_lr})')

        else:
            loss_without_hyperparam = loss_fn(output, target) + (lhpy + lhpz + lhp_diag)
            loss = (stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz + lhp_diag)) # consider hyperparameter sum division later

        loss_without_hyperparam_list.append(loss_without_hyperparam.data.cpu().numpy())
        loss_per_itr.append(loss.data.cpu().numpy())
        
        # loss_list.append(loss.data.cpu().numpy())

        loss.backward()
        optimizer.step()
    

    # %%
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ (c + 10) % nChannel ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( np.array([im.shape[0],im.shape[1],3]).astype( np.uint8 ))
    im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])

    # refine the cluster label, so that the cluster label is the same as the majority label of the pixels in the neighborhood
    def refine_cluster_label(im_cluster_num, im, radius = 1):
        im_cluster_num_refined = im_cluster_num.copy()
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if im_cluster_num[i, j] == 110: continue
                else:
                    cluster_label = im_cluster_num[i, j]
                    cluster_label_count = 0
                    other_cluster_labels = []
                    for k in range(-radius, radius + 1):
                        for l in range(-radius, radius + 1):
                            if i + k < 0 or i + k >= im.shape[0] or j + l < 0 or j + l >= im.shape[1]: continue
                            if im_cluster_num[i + k, j + l] == 110: continue
                            if im_cluster_num[i + k, j + l] == cluster_label: cluster_label_count += 1
                            else : other_cluster_labels.append(im_cluster_num[i + k, j + l])
                    if cluster_label_count < 4:
                        max_item,count = stats.mode(np.array(other_cluster_labels))
                        if len(count) != 0 and count[0] >= 4:
                            im_cluster_num_refined[i, j] = max_item[0]
        return im_cluster_num_refined
    im_cluster_num_refined = im_cluster_num.copy()
    for i in range(refine):
        im_cluster_num_refined = refine_cluster_label(im_cluster_num_refined, im, radius = 1)
    im_cluster_num = im_cluster_num_refined

    f = im_cluster_num
    s = np.argwhere(f != 110) # not a good way
    colors = f.flatten()
    plt.figure(figsize = (4, 4))
    if dataset == 'Custom': rad = 1500
    else: rad = 10
    plt.scatter(s[:, 1], 1000 - s[:, 0], c=colors, s = rad)

    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % nChannel ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( np.array([im.shape[0],im.shape[1],3]).astype( np.uint8 ))
    im_cluster_num = im_target.reshape(im.shape[0], im.shape[1])

    ## refinement
    im_cluster_num = im_cluster_num_refined

    labels = im_cluster_num[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]]
    grid_spots, colors = get_grid_spots_from_pixels(pixel_rows_cols, labels)

    df_ari_per_itr = pd.DataFrame({'ARI': ari_per_itr})
    df_ari_per_itr.to_csv(f'{leaf_output_folder_path}/ari_per_itr.csv')

    df_loss_per_itr = pd.DataFrame({'Loss': loss_per_itr})
    df_loss_per_itr.to_csv(f'{leaf_output_folder_path}/loss_per_itr.csv')

    df_loss_without_hyperparam_per_itr = pd.DataFrame({'Loss_without_hyperparam': loss_without_hyperparam_list})
    df_loss_without_hyperparam_per_itr.to_csv(f'{leaf_output_folder_path}/loss_without_hyperparam_per_itr.csv')

    df_labels = pd.DataFrame({'label': labels}, index=pixel_barcode[pixel_barcode != ''])
    df_labels.to_csv(f'{leaf_output_folder_path}/final_barcode_labels.csv')

    if sample == 'bcdc_ffpe' or sample == 'Melanoma':
        df_bayesSpace = pd.read_csv(f'./Data/others/{sample}/BayesSpace_output.csv')
        df_bayesSpace.set_index('Unnamed: 0', inplace=True)
        ari_bayesSpace = calc_ari(df_man, df_bayesSpace['spatial.cluster'])
        print('ari_bayesSpace', ari_bayesSpace)
        report_map['ari_bayesSpace'] = ari_bayesSpace

    if sample == 'Melanoma':
        # df_bayesSpace.set_index('Unnamed: 0', inplace=True)
        df_manual_partial = pd.read_csv('./Data/others/Melanoma/manual_annotations_wo_unannotated_reg.csv', index_col=0)
        ari_partial = calc_ari(df_manual_partial, df_labels.loc[df_manual_partial.index])
        ari_partial_bayesSpace = calc_ari(df_manual_partial, df_bayesSpace['spatial.cluster'].loc[df_manual_partial.index])
        print('ARI partial:', ari_partial)
        print('ARI partial BayesSpace:', ari_partial_bayesSpace)
        report_map['ari_partial'] = ari_partial
        report_map['ari_partial_bayesSpace'] = ari_partial_bayesSpace
    


    df_final_metrics = pd.DataFrame({'ARI': df_ari_per_itr['ARI'].values[-1:], 'Loss': df_loss_per_itr['Loss'].values[-1:], 'Loss_without_hyperparam': df_loss_without_hyperparam_per_itr['Loss_without_hyperparam'].values[-1:]})

    if sample == 'Melanoma':
        df_final_metrics['ARI partial'] = [ari_partial]
    
    df_final_metrics.to_csv(f'{leaf_output_folder_path}/final_metrics.csv')

    df_barcode_labels_per_itr.to_csv(f'{leaf_output_folder_path}/barcode_labels_per_itr.csv')

    print("ARI:", calc_ari(df_man, df_labels))
    if scribble:
        print(f"L_sim: {L_sim}, L_con: {L_con}, L_scr: {L_scr}")
        print(f"L_sim + L_con + L_scr: {L_sim + L_con + L_scr}")
        print(f"Total loss: {loss}")
        print(f"Loss without hyperparam: {loss_without_hyperparam}")


    report_map['ari'] = calc_ari(df_man, df_labels)

    

    if scribble:
        report_map['loss_sim'] = L_sim.data.cpu().numpy()
        report_map['loss_con'] = L_con.data.cpu().numpy()
        report_map['loss_scr'] = L_scr.data.cpu().numpy()
        report_map['loss_total'] = loss.data.cpu().numpy()

    meta_data_value = [test_name, seed, dataset, sample, n_pcs, scribble, max_iter, stepsize_sim, stepsize_con, stepsize_scr,alpha, scheme, lr, nConv, no_of_scribble_layers, intermediate_channels, added_layers, last_layer_channel_count, hyper_sum_division]
    df_meta_data = pd.DataFrame(index=meta_data_index, columns=['value'])
    df_meta_data['value'][meta_data_index] = meta_data_value
    df_meta_data.to_csv(meta_data_file_path)

    plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
    colors_to_plt = [plot_color[i%len(plot_color)] for i in labels]
    print("Last layer got: ",np.unique(labels).shape)

    if dataset == 'Custom': rad = 700
    elif sample == 'Melanoma': rad = 120
    else: rad = 10

    if sample == 'Melanoma': plt.figure(figsize=(1+5,5))
    else: plt.figure(figsize=(5,5))
    plt.axis('off')
    if sample != "Melanoma":
        plt.scatter(grid_spots[:, 1], 1000 - grid_spots[:, 0], c=colors_to_plt, s=rad)
    else:
        plt.scatter(grid_spots[:, 1], -1000 + grid_spots[:, 0], c=colors_to_plt, s=rad)
    # plt.savefig(f'{leaf_output_folder_path}/seg_{stepsize_sim}_{stepsize_con}_{stepsize_scr}_seed_{seed}_pcs_{n_pcs}.png',format='png',dpi=1200,bbox_inches='tight',pad_inches=0)
    # plt.savefig(f'{leaf_output_folder_path}/seg_{stepsize_sim}_{stepsize_con}_{stepsize_scr}_seed_{seed}_pcs_{n_pcs}.eps',format='eps',dpi=1200,bbox_inches='tight',pad_inches=0)
    plt.savefig(f'{leaf_output_folder_path}/seg_{alpha}.png',format='png',dpi=1200,bbox_inches='tight',pad_inches=0)
    plt.savefig(f'{leaf_output_folder_path}/seg_{alpha}.eps',format='eps',dpi=1200,bbox_inches='tight',pad_inches=0)

    plt.close('all')

    # print(report_map)
    report_list.append(report_map)

report_dataf = pd.DataFrame(report_list)
report_dataf.to_csv(f'Report/report_{dataset}.csv')
