import streamlit as st
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from matplotlib.image import imread
import time
import torch
import torchvision
from torchmetrics import StructuralSimilarityIndexMeasure
def Bytes(Bytes) :
    kilobytes = Bytes / 1024
    return kilobytes


def load_image(image_file):
	img = Image.open(image_file)
	return img


st.subheader("Image")
filename = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

last_time = time.time()
if filename is not None:

    A=np.asarray(load_image(filename))
    Red = A[:,:,0]
    Green = A[:,:,1]
    Blue  = A[:,:,2]

    info_list = []
    for i in np.arange(0.005,0.04,0.010): 

        final_list = []
        for color in (Red, Green, Blue) :  
            
            fft_color = torch.fft.fft2(torch.Tensor(color))
            sorted_fft_color = np.sort(np.abs(fft_color.reshape(-1))) 
            
            keep=torch.from_numpy(np.asarray(i))
            thresh = sorted_fft_color[int(torch.floor((1-keep)*len(sorted_fft_color)))-1]
            filter_ = torch.abs(fft_color)>thresh         
            filtered_fft = fft_color * filter_   

            inverse_fft = torch.fft.ifft2(filtered_fft).real
            final_list.append(inverse_fft.cpu().detach().numpy())

        modified = np.dstack((final_list[0],final_list[1],final_list[2])).astype('uint8')
        score = []
    
        for j in range(3):
            dim_orig = A[:,:,j]
            modified_orig = modified[:,:,j]
            score.append(StructuralSimilarityIndexMeasure(dim_orig, modified_orig))
        
        new_path = './compressed_images/'+filename.name.split('.')[0]+'_'+str(i)[:5]+'.jpg'

        Image.fromarray(modified).save(new_path)
        image_file = Image.open(new_path)
    
        info_list.append((i,sum(score)/3,Bytes(len(image_file.fp.read())),Bytes(filename.size)))

    temp_list = [i for i in info_list if i[2]<i[3]]

    if temp_list is not None:
        new_list = temp_list[int(len(temp_list)/3)]
    else:
        new_list = [info_list[0]]

    percentage = (new_list[0]/info_list[-1][0])*100

    st.write('Percentage:' ,percentage)
    st.write('Size Before Compression (KB): ' ,new_list[3])
    st.write('Size After Compression (KB): ' ,new_list[2])
    

    keep=new_list[0]
    print(keep)
    thresh = sorted_fft_color[(int(np.floor((1-keep)*len(sorted_fft_color))))-1]

    filter_ = np.abs(fft_color)>thresh         
    filtered_fft = fft_color * filter_   

    inverse_fft = np.fft.ifft2(filtered_fft).real
    final_list.append(inverse_fft)

    modified = np.dstack((final_list[0],final_list[1],final_list[2])).astype('uint8')
    
    col1, col2 = st.columns(2)
        
    with col1:
        st.header("Compressed Version"+str(np.asarray(Image.fromarray(modified)).shape))
        st.image(Image.fromarray(modified), width=250)

    with col2:
        st.header("Original Version"+str(np.asarray(load_image(filename)).shape))
        st.image(load_image(filename), width=250)
    print(last_time - time.time())
