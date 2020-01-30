
#...............................Imports..................................................................

import os
import torch
import numpy as np
import FCN_NetModel as FCN # The net Class
import CategoryDictionary as CatDic
import cv2
#import scipy.misc as misc

############################################Input parameters###################################################################################
#-------------------------------------Input parameters-----------------------------------------------------------------------
InputDir="InputImages/" # Folder of input images
OutDir="Out/" # Folder of output

UseGPU=False # Use GPU or CPU  for prediction (GPU faster but demend nvidia GPU and CUDA installed else set UseGPU to False)
FreezeBatchNormStatistics=False # wether to freeze the batch statics on prediction  setting this true or false might change the prediction mostly False work better
OutEnding="" # Add This to file name
if not os.path.exists(OutDir): os.makedirs(OutDir) # Create folder for trained weight

#-----------------------------------------Location of the pretrain model-----------------------------------------------------------------------------------
Trained_model_path ="logs//TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"
##################################Load net###########################################################################################
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=FCN.Net(CatDic.CatNum) # Create net and load pretrained encoder path
if UseGPU==True:
    print("USING GPU")
    Net.load_state_dict(torch.load(Trained_model_path))
else:
    print("USING CPU")
    Net.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))
#--------------------------------------------------------------------------------------------------------------------------
# Net.half()
for name in os.listdir(InputDir): # Main read and predict results for all files


#..................Read and resize image...............................................................................
    print(name)
    InPath=InputDir+"/"+name
    Im=cv2.imread(InPath)
    h,w,d=Im.shape
    r=np.max([h,w])
    if r>840: # Image larger then 840X840 are shrinked (this is not essential, but the net results might degrade when using to large images
        fr=840/r
        Im=cv2.resize(Im,(int(w*fr),int(h*fr)))
    Imgs=np.expand_dims(Im,axis=0)
    if not (type(Im) is np.ndarray): continue
#................................Make Prediction.............................................................................................................
    with torch.autograd.no_grad():
          OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNormStatistics) # Run net inference and get prediction
#...............................Save prediction on fil
    print("Saving output to: " + OutDir)
    for nm in OutLbDict:
        Lb=OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8)
        if Lb.mean()<0.001: continue
        if nm=='Ignore': continue
        ImOverlay1 = Im.copy()
        ImOverlay1[:, :, 0][Lb==1] = 255
        ImOverlay1[:, :, 1][Lb==1] = 0
        ImOverlay1[:, :, 2][Lb==1] = 255
        FinIm=np.concatenate([Im,ImOverlay1],axis=1)
        OutPath = OutDir + "//" + nm+"/"

        if not os.path.exists(OutPath): os.makedirs(OutPath)
        OutName=OutPath+name[:-4]+OutEnding+".png"
        cv2.imwrite(OutName,FinIm)







