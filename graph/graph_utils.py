import pandas as pd
from glob import glob
import cv2
import numpy as np
import os
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.figure_factory as ff
cityscapes_trainIds= {
       0: 'road'          ,    
       1: 'sidewalk'      ,  
       2: 'building'      ,  
       3: 'wall'          ,  
       4: 'fence'         ,  
       5: 'pole'          ,  
       6: 'traffic light' ,  
       7: 'traffic sign'  ,  
       8: 'vegetation'    ,  
       9: 'terrain'       ,  
      10: 'sky'           ,  
      11: 'person'        ,  
      12: 'rider'         ,  
      13: 'car'           ,  
      14: 'truck'         ,  
      15: 'bus'           ,  
      16: 'train'         ,  
      17: 'motorcycle'    ,  
      18: 'bicycle'       ,
      19: 'ABSTAIN'}

cityscapes_catIds = {
0: 'flat'         ,  
1: 'construction' ,  
2: 'object'       ,   
3: 'nature'       ,   
4: 'sky'          ,
5: 'human'        ,
6: 'vehicle'      ,
7: 'ABSTAIN'    }
param_split_dict = {'n':3, 'n0':4, 'tau':5}


def graph_gifs(log_dir, param):

    vid_types = ['certify', 'pvals']
    for vid_type in vid_types:
        os.makedirs(f'{log_dir}/gifs_{vid_type}_{param}', exist_ok=True)
        os.makedirs(f'{log_dir}/annotated_{vid_type}', exist_ok=True)
        print(f'{log_dir}/annotated_{vid_type}')

    paths = glob(f'{log_dir}/*.png')

    df = pd.read_csv(f'{log_dir}/images_df.csv')
    param_split_dict = {'n':3, 'n0':4, 'tau':5}
    indices_ls = []
    


    for image_pth in paths:
        im_name = image_pth.split('/')[-1]
        for vid_type in vid_types:
            if im_name.find(vid_type) >=0:
                image = cv2.imread(image_pth)
                font = cv2.FONT_HERSHEY_SIMPLEX

                top = int(0.1 * image.shape[0])  # shape[0] = rows
                image = cv2.copyMakeBorder(image, top, 0, 0, 0, cv2.BORDER_CONSTANT, None, (255, 255, 255))
                
                
                split_name = im_name[:-4].split('_')
                index = int(split_name[0])
                correction = split_name[2]
                n = int(split_name[3])
                n0 = int(split_name[4])
                tau = float(split_name[5])
                #if n0 > n: continue
                row_condition = (df['idx'] == index) & (df['tau'] == tau) & (df['n'] == n) & (df['n0'] == n0) & (df['correction'] == correction)
                row = df[row_condition]

                tau = round(tau, 2)

                if len(row['non_abstain'] > 0):
                    non_abstain = row['non_abstain'].values[0].round(2)
                    accuracy = row['accuracy'].values[0].round(2)
                    N = int(row['N'].values[0])
                    radius = row['radius'].values[0].round(2)
                    if param =='n':
                        text_on_image = f'n={n}, %certified={non_abstain}, accuracy={accuracy}   (R={radius}, N={N}, n0={n0}, tau={tau})'
                    elif param =='n0':
                        text_on_image = f'n0={n0}, %certified={non_abstain}, accuracy={accuracy}              (n={n}, tau={tau}, {correction})'
                    elif param == 'tau':
                        text_on_image = f'tau={tau}, %certified={non_abstain}, accuracy={accuracy}              (n={n}, n0={n0}, {correction})'


                    # Using cv2.putText() method
                else:
                    continue
                    #text_on_image = f'N={eval('N')}                              (n={n}, n0={n0}, tau={tau})'

                font =  cv2.FONT_HERSHEY_SIMPLEX
                org = (20, 70)
                fontScale = 1.5
                color = (0, 0, 0)
                thickness = 2

                image = cv2.putText(image, text_on_image, org, font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.imwrite(f'{log_dir}/annotated_{vid_type}/{im_name}', image)

                indices_ls.append(int(index))

    histogram = np.bincount(indices_ls)
    corr = ['holm', 'bonferroni']

    # per every index, make 2 gifs (for every correction):
    for idx in set(indices_ls):
        for c in corr:
            for vid_type in vid_types:
                images = sorted(glob(f'{log_dir}/annotated_{vid_type}/{idx}_{vid_type}_{c}_*.png'), key=lambda x: float(x.split('/')[-1][:-4].split('_')[param_split_dict['n']])+float(x.split('/')[-1][:-4].split('_')[param_split_dict['n0']] ))
                if len(images) <= 0: continue
                height, width, _ = cv2.imread(images[0]).shape
                out = cv2.VideoWriter(f'{log_dir}/gifs_{vid_type}_{param}/{idx}_{c}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
                for im in images:
                    img = cv2.imread(im)
                    out.write(img)
                out.release()

def graph_param1_vs_param2(log_dir, param1, param2, to_title_dict, x_title=None, corr=['holm', 'bonferroni']):

    df = pd.read_csv(f'{log_dir}/images_df.csv')

    df = df[['idx', param1, param2, 'correction']]
    indices = np.unique(df['idx'])
    for idx in indices:
        for c in corr:
            sub_df = df[(df['idx'] == idx) & (df['correction'] == c)]
            param1_values = sub_df[param1].values
            param2_values = sub_df[param2].values
            graph_scatter(x=[param2_values], y=[param1_values], x_title=param2,
                    y_title=to_title_dict[param1], labels=[' '],
                    graph_title=f'{to_title_dict[param1]} vs. {to_title_dict[param2]} ({c})',
                    graphs_folder_name=f'{log_dir}/graphs/{param1}_{param2}',
                    graph_file_name=f'{idx}_{c}')

def graph_histogram(log_dir, param='n'):
    log_dir_ = log_dir
    os.makedirs(f'{log_dir}/histograms/gifs', exist_ok=True)
    os.makedirs(f'{log_dir}/boundary_histogram/gifs', exist_ok=True)

    df = pd.read_csv(f'{log_dir}/images_df.csv')
    df = df.reset_index()

    indices_ls = []
    for index, row in df.iterrows():
        idx = row['idx']

        if pd.isna(row['abstain_histogram']) or pd.isna(row['abstain_histogram_boundary_normalized']):
            continue
        abstain_histogram = list(str_ls_to_array(row['abstain_histogram']))
        boundary_histogram = list(eval(row['abstain_histogram_boundary_normalized']))
        hierarchical = bool(row['hierarchical'])
        if hierarchical:
            t_dict = cityscapes_catIds
        else:
            t_dict = cityscapes_trainIds
        #confusion_matrix = str_to_array(confusion_matrix)
        labels = [t_dict[i] for i in range(len(abstain_histogram))]
        labels_boundary = ['Non-boundary', 'Boundary']
        n, n0, tau, accuracy, non_abstain, correction = int(row['n']), int(row['n0']),\
                        round(row['tau'],2), round(row['accuracy'], 2),\
                        round(row['non_abstain'], 2), row['correction']

        if param =='n':
            text_on_image = f'n={n}, %certified={non_abstain}, accuracy={accuracy}'
        elif param =='n0':
            text_on_image = f'n0={n0}, %certified={non_abstain}, accuracy={accuracy}'
        elif param == 'tau':
            text_on_image = f'tau={tau}, %certified={non_abstain}, accuracy={accuracy}'
        graph_bar([labels], [abstain_histogram], 'Class Labels', 'Number of Abstain Pixels',\
            ['Abstain Count'], f'Histogram of Abstain Pixels {text_on_image}', f'{log_dir}/histograms', f'{idx}_certify_{correction}_{n}_{n0}_{tau}')

        graph_bar([labels_boundary], [boundary_histogram], 'Pixel Type', 'Percentage of Abstained Pixels',\
            ['Abstain Percentage'], f'Relative Percentage of Abstain Pixels {text_on_image}', f'{log_dir}/boundary_histogram', f'{idx}_certify_{correction}_{n}_{n0}_{tau}')

        indices_ls.append(idx)
    corr = ['holm', 'bonferroni']
    # per every index, make 2 gifs (for every correction):
    for idx in tqdm(set(indices_ls), desc='Creating GIFs for Histograms'):
        for c in corr:
            images = sorted(glob(f'{log_dir}/histograms/{idx}_certify_{c}_*.png'), key=lambda x: float(x.split('/')[-1][:-4].split('_')[param_split_dict[param]]))
            if len(images) <= 0: continue
            height, width, _ = cv2.imread(images[0]).shape
            out = cv2.VideoWriter(f'{log_dir}/histograms/gifs/{idx}_{c}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
            for im in images:
                img = cv2.imread(im)
                out.write(img)
            out.release() 

    for idx in tqdm(set(indices_ls), desc='Creating GIFs for Boundary Histograms'):
        for c in corr:
            images = sorted(glob(f'{log_dir}/boundary_histogram/{idx}_certify_{c}_*.png'), key=lambda x: float(x.split('/')[-1][:-4].split('_')[param_split_dict[param]]))
            if len(images) <= 0: continue
            height, width, _ = cv2.imread(images[0]).shape
            out = cv2.VideoWriter(f'{log_dir}/boundary_histogram/gifs/{idx}_{c}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
            for im in images:
                img = cv2.imread(im)
                out.write(img)
            out.release() 
def graph_boundary_histogram_for_all(setups, exp_dirs):
    hists = {}
    df = pd.read_csv(f'{exp_dirs[0]}/images_df.csv')

    combined_df = pd.DataFrame(columns=df.columns)
    indices = []
    for dir, setup in zip(exp_dirs, setups):
        df = pd.read_csv(f'{dir}/images_df.csv')
        df = df[~df['baseline']]
        df = df[df.groupby(['name'])['accuracy'].transform(max) == df['accuracy']]
        hists[setup] = df
        indices.append(df['idx'].values)
    num_images = len(df)

    indices = list(set([y for x in indices for y in x]))

    for idx in indices:
        hists_per_setup = []
        for setup in setups:
            df = hists[setup]
            hists_per_setup.append(str_ls_to_array(df['abstain_histogram_boundary_normalized'][df['idx'] == idx].values[0]))

        labels_boundary = ['Non-boundary', 'Boundary']

        graph_bar([labels_boundary for i in range(len(setups))], hists_per_setup, 'Pixel Type', 'Percentage of Abstained Pixels',\
            setups, f'Relative Percentage of Abstain Pixels', f'/home/alaa/Academics/certification/segmentation-smoothing/logs/boundary_histogram', f'{idx}')






def graph_bar_params_vs_param1(log_dir, params, param2, to_title_dict, y_title = 'Time (s)', barmode='stack', corr=['holm', 'bonferroni']):

    df = pd.read_csv(f'{log_dir}/images_df.csv')
    cols = ['idx', 'correction', param2] + [param for param in  params]
    df = df[cols]
    indices = np.unique(df['idx'])
    for idx in indices:
        for c in corr:
            sub_df = df[(df['idx'] == idx) & (df['correction'] == c)]
            ys = [sub_df[param].values for param in params]
            param2_values = sub_df[param2].values
            xs = [param2_values for i in range(len(ys))]
            graph_bar(x=xs, y=ys, x_title=to_title_dict[param2],
                    y_title='Time (s)', labels=params,
                    graph_title=f'{y_title} vs. {to_title_dict[param2]} ({c})',
                    graphs_folder_name=f'{log_dir}/graphs/{y_title}_{param2}',
                    graph_file_name=f'{idx}_{c}',
                    barmode=barmode)
def str_ls_to_array(ls):
    conf = []
    for x in ls.split(' '):
        x = x.replace('[', '')
        x = x.replace(']', '')
        x = x.replace(',', '')

        if len(x) == 0: continue
        conf.append(float(x))
    length = len(conf)
    return np.array(conf).reshape(length)
def str_to_array(confusion_matrix):
    conf = []
    for x in confusion_matrix.split(' '):
        x = x.replace('[', '')
        x = x.replace(']', '')
        if len(x) == 0: continue
        conf.append(int(float(x))) 
    num_classes = int(np.sqrt(len(conf)))
    return np.array(conf).reshape((num_classes, num_classes))

def graph_confusion_matrix(log_dir, params = ['n', 'n0', 'tau']):
    log_dir_ = log_dir
    for param in tqdm(params, desc='graph_confusion_matrix'):
        log_dir = f'{log_dir_}/{param}'
        os.makedirs(f'{log_dir}/confusion_matrix/gifs', exist_ok=True)

        df = pd.read_csv(f'{log_dir}/images_df.csv')
        df = df.reset_index()

        indices_ls = []
        for index, row in df.iterrows():
            idx = row['idx']
            baseline = row['baseline']
            confusion_matrix = row['confusion_matrix']
            if pd.isna(confusion_matrix):
                continue
            confusion_matrix = str_to_array(confusion_matrix)
            labels = [cityscapes_trainIds[i] for i in range(confusion_matrix.shape[0])]
            if baseline:
                graph_title = 'Baseline Confusion Matrix'
                graph_heatmap(confusion_matrix, labels, graphs_folder_name=f'{log_dir}/confusion_matrix', 
                        graph_file_name=f'{idx}_baseline', graph_title=graph_title, write=1)
            else:
                n, n0, tau, accuracy, non_abstain, correction = int(row['n']), int(row['n0']),\
                             round(row['tau'],2), round(row['accuracy'], 2),\
                             round(row['non_abstain'], 2), row['correction']

                if param =='n':
                    text_on_image = f'n={n}, %certified={non_abstain}, accuracy={accuracy}   (n0={n0}, tau={tau}, {correction})'
                elif param =='n0':
                    text_on_image = f'n0={n0}, %certified={non_abstain}, accuracy={accuracy}   (n={n}, tau={tau}, {correction})'
                elif param == 'tau':
                    text_on_image = f'tau={tau}, %certified={non_abstain}, accuracy={accuracy}   (n={n}, n0={n0}, {correction})'
                graph_heatmap(confusion_matrix, labels, graphs_folder_name=f'{log_dir}/confusion_matrix', 
                        graph_file_name=f'{idx}_certify_{correction}_{n}_{n0}_{tau}', graph_title=text_on_image, write=1)
            
            indices_ls.append(idx)
        corr = ['holm', 'bonferroni']
        # per every index, make 2 gifs (for every correction):
        for idx in tqdm(set(indices_ls), desc='Creating GIFs for Confusion Matrix'):
            for c in corr:
                images = sorted(glob(f'{log_dir}/confusion_matrix/{idx}_certify_{c}_*.png'), key=lambda x: float(x.split('/')[-1][:-4].split('_')[param_split_dict[param]]))
                if len(images) <= 0: continue
                height, width, _ = cv2.imread(images[0]).shape
                out = cv2.VideoWriter(f'{log_dir}/confusion_matrix/gifs/{idx}_{c}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
                for im in images:
                    img = cv2.imread(im)
                    out.write(img)
                out.release()
        
            

def graph_heatmap(conf_matrix, labels, graphs_folder_name, graph_file_name, graph_title, write=1):

    # change each element of z to type string for annotations
    conf_matrix_text = [[str(y) for y in x] for x in conf_matrix]
    # set up figure 

    fig = go.Figure(data=go.Heatmap(z=conf_matrix, text=conf_matrix_text,
                    x=labels, y=labels, colorscale='Viridis'))
    fig.update_layout(
        title=graph_title,
        xaxis_title='Predicted Value',
        yaxis_title = 'Ground Truth'
    )
    # add colorbar
    os.makedirs(graphs_folder_name, exist_ok=True)
    if write:
        fig.write_image(f'{graphs_folder_name}/{graph_file_name}.png')
        fig.write_html(f'{graphs_folder_name}/{graph_file_name}.html')


def graph_bar(x, y, x_title, y_title,\
    labels, graph_title, graphs_folder_name, graph_file_name, barmode='group', fig2=None,\
    annotation=None, write=1):
    os.makedirs(graphs_folder_name, exist_ok=True)
    fig = go.Figure()
    data = []
    for x_, y_, label in zip(x, y, labels):
        fig.add_trace(go.Bar(name=label, x=x_, y=y_))
    # Change the bar mode
    fig.update_layout(barmode=barmode)

    fig.update_layout(
        title=graph_title,\
        xaxis_title=x_title,\
        yaxis_title=y_title
    )
    if write:
        fig.write_image(f'{graphs_folder_name}/{graph_file_name}.png')
        fig.write_html(f'{graphs_folder_name}/{graph_file_name}.html')
    return fig
# Graphing Cell
def graph_scatter(x, y, x_title, y_title,\
    labels, graph_title, graphs_folder_name, graph_file_name, fig2=None,\
    annotation=None, write=1):

    if not os.path.exists(graphs_folder_name):
        os.makedirs(graphs_folder_name)
    if fig2 is None:
        fig = go.Figure()
    else:
        fig = fig2

    for xi, yi, label in zip(x, y, labels):
        fig.add_trace(go.Scatter(x=xi, y=yi, name=label,
                        mode='lines+markers'))

    fig.update_layout(
        title=graph_title,\
        xaxis_title=x_title,\
        yaxis_title=y_title
    )
    if annotation is not None:
        for ann in annotation:
            fig.add_annotation(
                    x=ann[0],
                    y=ann[1],
                    xref="x",
                    yref="y",
                    text=ann[2],
                    showarrow=True,
                    font=dict(
                        family="Courier New, monospace",
                        size=13,
                        color="#000000"
                        ),
                    align="center",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=20,
                    ay=-40,
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=3,
                    bgcolor="#ff7f0e",
                    opacity=0.8
                    )

    fig.update_xaxes(fixedrange=True)
    if write:
        fig.write_image(f'{graphs_folder_name}/{graph_file_name}.png')
        fig.write_html(f'{graphs_folder_name}/{graph_file_name}.html')
    return fig


def shift_frame(img,move_dir,fill=np.inf):
    frame = np.full_like(img,fill)
    x,y = move_dir
    size_x,size_y = np.array(img.shape) - np.abs(move_dir)
    frame_x = slice(0,size_x) if x>=0 else slice(-x,size_x-x)
    frame_y = slice(0,size_y) if y>=0 else slice(-y,size_y-y)
    img_x = slice(x,None) if x>=0 else slice(0,size_x)
    img_y = slice(y,None) if y>=0 else slice(0,size_y)
    frame[:, frame_x,frame_y] = img[:, img_x,img_y]
    return frame
