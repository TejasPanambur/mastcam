import torch
import torch.nn as nn
import torchvision.datasets as datasets
import os
import models
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
from glob import glob
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd 
import torchvision.transforms as transforms
###### Loading the model
import pickle
import random
np.random.seed(0)

global col
random.seed(0)
col = np.array([[random.randrange(0, 255)/255, random.randrange(0, 255)/255, random.randrange(0,255)/255] for i in range(78)])
print(col[0].shape)

def load_model(resume,arch='texres18'):
    model = models.__dict__[arch](sobel=False,bn=False)
    #fd = int(model.top_layer.weight.size()[1])
    #model.top_layer = None
    model.features = torch.nn.DataParallel(model.features).to('cuda:0')
    #model.to('cpu')
    cudnn.benchmark = True
    #print(torch.cuda.device_count())
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume,map_location='cpu')
        start_epoch = checkpoint['epoch']
        # remove top_layer parameters from checkpoint
        for key in list(checkpoint['state_dict']):
            if 'top_layer' in key:
                del checkpoint['state_dict'][key]
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
	  .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
    return model

####Get the csv data
def getDF(pth,epoch=None):
    df = pd.read_csv(pth)
    df['ImageFolder'] = df.apply(lambda x: x['ImageName'].split('/')[-2],axis=1)
    imgpth='/mnt/nfs/work1/mahadeva/mparente/dataset/texture/'
    df['ImageName'] = df.apply(lambda x:x['ImageName'].replace('/local/',imgpth),axis=1)
    df_keys = list(df.keys())
    if epoch==None:
        epoch = 0
        for k in df_keys:
            try:
                epoch = max(epoch,int(k))
            except:
                pass
    #df = df[['ImageName','ImageFolder',str(epoch)]].rename(columns={str(epoch):'class'})
    df['class_lbl'] = df.apply(lambda x: x['ImageName'].split('/')[-3],axis=1)
    labelencoder = LabelEncoder()
    df['true_lbl'] = labelencoder.fit_transform(df['class_lbl'])
    df = df.rename(columns={str(epoch):'class'})#[['ImageName','ImageFolder',str(epoch),'true_lbl']].rename(columns={str(epoch):'class'})   
    return df


def get_imgpth_lbls(df,imgfolder):
    imgpths = df[df['ImageFolder']==imgfolder]['ImageName'].to_list()
    lbls = df[df['ImageFolder']==imgfolder]['class'].to_list()
    return imgpths,lbls
    
#### Transform the Image
def transf(img):
    tra = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),
    transforms.Normalize(mean = [122.7717, 115.9465, 102.9801], std = [1, 1, 1]),
    transforms.Lambda(lambda x: x[[2, 1, 0], ...])
    ])
    res = tra(img)
    return res


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

###Get the images and its transforms
def get_npimg_transimg(imgpths):
    img_list = []

    for i, imgp in enumerate(imgpths):
        imgpth,c = imgp[0],imgp[1]
        #print(imgpth)
        im = Image.open(imgpth)
        img_list.append(cv2.rectangle(np.array(im.resize((256,256))),(0,0),(256,256),col[c]*255,30))          
        timg = transf(im)
        if i == 0:
            input_var = torch.zeros((len(imgpths), timg.shape[0], timg.shape[1], timg.shape[2]))		
        input_var[i]=timg
    return img_list,input_var	

####Get embeds
def getembeds(model,input_var,batch=32):
    model.eval()
    out = np.ones((len(input_var),512),dtype='float32')
    with torch.no_grad():
        for i in range(0,len(input_var),batch):
            print('i',i)
            print('sha',input_var[i:i+batch].shape)
            if i+batch < len(input_var) - 1:       
                out[i:i+batch] =  model(input_var[i:i+batch].float()).data.cpu().numpy()  
                     
                #print(model(input_var[i:i+batch].float()).data.cpu().numpy().shape)
            else:
                # special treatment for final batch
                out[i:] = model(input_var[i:]).data.cpu().numpy()
    print(out.shape)
    return out

####Get TSNE
def getTSNEembed(embed,ee=12.0,perp=40.0):
    tsne = TSNE(n_components=2, early_exaggeration=ee,perplexity=perp,init='pca')
    tsne_result = tsne.fit_transform(embed)
    tsne_result = StandardScaler().fit_transform(tsne_result)
    return tsne_result

#### Plot Scatter
def visualize_scatter(data_2d, label_ids, path, figsize=(20,20)):
    print('Visualizing Scatter')
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    c= np.array([col[label_id]]),
                    linewidth=1,
                    alpha=0.9,
                    label=label_id)
    plt.legend(loc='best')
    plt.savefig(path+'_scatter.png')
    

####Visualize with images
def visualize_scatter_with_images(X_2d_data, images, path, figsize=(50,50), image_zoom=0.6):
    print('Visualizing scatter with images')
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    #plt.show()
    plt.savefig(path+'_img_scatter.png')


def compute():
    workpath ='results/textureN/gtosf_11_hyp_inception_xbm_triplet_150Reg/'
    CP = workpath+'checkpoints/checkpoint.pth.tar'
    dfPth = workpath+'clustered.csv'
    respth = 'plots/'
    model = load_model(False)
    df = getDF(dfPth)
    print('Obtained df')
    imgpths = []
    lbls = []
    imgfolders = set(df['ImageFolder'])
    '''
    for imgfolder in imgfolders:
    print('Computing ', imgfolder)
    imgpth,lbl = get_imgpth_lbls(df,imgfolder)
    imgpths += imgpth
    lbls += lbl'''
    imgpths = df['ImageName'].to_list()
    lbls = df['true_lbl'].to_list()
    #print(imgpths[0])
    imgpths,lbls = imgpths[::2],lbls[::2]       
    #print(imgpths[0])
    #print(len(set(lbls)))
    imgpths = [[imgpths[i],lbls[i]] for i in range(len(imgpths))]
    '''with open ('imagenames', 'rb') as fp:
    imgpths = pickle.load(fp)'''
    #imgpths = ['../dataset/'+i.split('/local/')[1] for i in imgpths]
    #imgpths = imgpths[::3]
    img_list,input_var = get_npimg_transimg(imgpths)
    #print(len(img_list))
    #print('Obtained trans Images')
    print('Img',input_var)
    inp_embed = getembeds(model,input_var)
    print('embed',inp_embed)
    tsne_embed = getTSNEembed(inp_embed, ee=12.0,perp=40)
    #visualize_scatter(tsne_embed, lbls, respth+'gtos_all_0')
    #visualize_scatter(tsne_embed, df['0'].to_list(), respth+'gtos_all_ps_0')
    visualize_scatter(tsne_embed, lbls, respth+'e0_gtos_all_true_0')
    tsne_embed = getTSNEembed(inp_embed, ee=12.0,perp=50)
    visualize_scatter(tsne_embed,lbls,respth+'e0_gtos_all_true_perp_50')
    tsne_embed = getTSNEembed(inp_embed, ee=20.0,perp=50)
    visualize_scatter(tsne_embed,lbls,respth+'e0_gtos_all_true_ee_20')
    tsne_embed = getTSNEembed(inp_embed, ee=40.0,perp=50)
    visualize_scatter(tsne_embed,lbls,respth+'e0_gtos_all_true_ee_40')
    tsne_embed = getTSNEembed(inp_embed, ee=200.0,perp=60)
    visualize_scatter(tsne_embed, lbls, respth+'e0_gtos_all_true_perp_90')
    #print(tsne_embed)
    #visualize_scatter_with_images(tsne_embed, img_list, respth+'gtos_all_0')

#compute()

