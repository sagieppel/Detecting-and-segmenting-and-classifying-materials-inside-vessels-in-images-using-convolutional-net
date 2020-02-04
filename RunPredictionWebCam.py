#Run  net on webcam to overlay  prediction image  on and display  on screen
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
#InputVideo=r"C:\Users\Sagi\Documents\2020.1.28_lab_captures\1580404054104.mp4" # Input Video
#OutVideoMain=InputVideo[:-4]+"_MainClasses.avi" #Output video that contain vessel filled  liquid and solid
#OutVideoAll=InputVideo[:-4]+"_AllClasses.avi"#Output video that contain subclasses that have more then 5% of the image
UseGPU=False # Use GPU or CPU  for prediction (GPU faster but demend nvidia GPU and CUDA installed else set UseGPU to False)
FreezeBatchNormStatistics=False # wether to freeze the batch statics on prediction  setting this true or false might change the prediction mostly False work better
OutEnding="" # Add This to file name

#-----------------------------------------Location of the pretrain model-----------------------------------------------------------------------------------
Trained_model_path =r"logs//TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"

##################################Load net###########################################################################################
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=FCN.Net(CatDic.CatNum) # Create net and load pretrained encoder path
if UseGPU==True:
    print("USING GPU")
    Net.load_state_dict(torch.load(Trained_model_path))
else:
    print("USING CPU")
    Net.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))
#---------------------OPEN video-----------------------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
MainCatsVideoWriter=None
AllCatsVideoWriter=None
#--------------------Create output video---------------------------------------------------------------------------------


#-----------------------Read Frame one by one-----------------------------------------------------------------------
# Read until video is completed
#iii=0
while (cap.isOpened()):
   # if iii>3: break
    # Capture frame-by-frame
    # ..................Read and resize image...............................................................................

    ret, Im = cap.read()
    if ret == False: continue
        # Display the resulting frame

    h,w,d=Im.shape
    r=np.max([h,w])
    if r>840: # Image larger then 840X840 are shrinked (this is not essential, but the net results might degrade when using to large images
        fr=840/r
        Im=cv2.resize(Im,(int(w*fr),int(h*fr)))
    h, w, d = Im.shape
    Imgs=np.expand_dims(Im,axis=0)
    if not (type(Im) is np.ndarray): continue
#................................Make Prediction.............................................................................................................
    with torch.autograd.no_grad():
          OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNormStatistics) # Run net inference and get prediction
#------------------------------------Display main classes on the image----------------------------------------------------------------------------------

    my=2
    mx=2
    OutMain = np.zeros([h * my, w * mx, 3], np.uint8)
    y = 0
    x = 0
    OutMain[:h,:w]=Im
    MainCatName = ['Vessel','Filled','Liquid GENERAL','Solid GENERAL']
    VesMat = OutLbDict['Vessel'].data.cpu().numpy()[0].astype(np.uint8)
    for nm in MainCatName:
        Lb=OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8)
        #if Lb.mean()<0.001: continue
        if nm=='Ignore': continue
        font = cv2.FONT_HERSHEY_SIMPLEX

        ImOverlay1 = Im.copy()
        ImOverlay1[:, :, 1][Lb==1] = 0
        cv2.putText(ImOverlay1, nm, ( int(w/3), int(h/6)), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        ImOverlay1[:, :, 0][Lb==1] = 255
        OutMain[h*y:h*(y+1), w*x:w*(x+1)] = ImOverlay1
        x+=1
        if x>=mx:
            x=0
            y+=1
    h,w,d=OutMain.shape
    r = np.max([h, w])
    if r>1000: # Image larger then 840X840 are shrinked (this is not essential, but the net results might degrade when using to large images
        fr=1000/r
        OutMain=cv2.resize(OutMain,(int(w*fr),int(h*fr)))
    h, w, d = OutMain.shape
    cv2.imshow('Main Classes', OutMain)
    cv2.waitKey(25)
    # if MainCatsVideoWriter is None:
    #     h, w, d = OutMain.shape
    #     MainCatsVideoWriter = cv2.VideoWriter(OutVideoMain, fourcc, 20.0, (w, h))
    # MainCatsVideoWriter.write(OutMain)
#------------------------------------Display all classes on the image----------------------------------------------------------------------------------
    h, w, d = Im.shape
    my=3
    mx=3
    OutMain = np.zeros([h * my, w * mx, 3], np.uint8)
    y = 0
    x = 1
    OutMain[:h,:w]=Im
    AllCatName = ['Vessel','Liquid GENERAL','Solid GENERAL','Foam','Powder','Liquid Suspension','Granular','V Label','V Cork','Gel','Solid Bulk','Vapor']

    VesMat = OutLbDict['Vessel'].data.cpu().numpy()[0].astype(np.uint8)
    for nm in AllCatName:
        Lb=OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8)
        if Lb.mean()<0.0005: continue
        if nm=='Ignore': continue
        font = cv2.FONT_HERSHEY_SIMPLEX

        ImOverlay1 = Im.copy()
        ImOverlay1[:, :, 1][Lb==1] = 0
        cv2.putText(ImOverlay1, nm, ( int(w/3), int(h/6)), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        ImOverlay1[:, :, 0][Lb==1] = 255
        OutMain[h*y:h*(y+1), w*x:w*(x+1)] = ImOverlay1
        x+=1
        if x>=mx:
            x=0
            y+=1
        if y>2: break
    h,w,d=OutMain.shape
    r = np.max([h, w])
    if r>1800: # Image larger then 840X840 are shrinked (this is not essential, but the net results might degrade when using to large images
        fr=1800/r
        OutMain=cv2.resize(OutMain,(int(w*fr),int(h*fr)))
  #  cv2.imshow('All Classes', OutMain)
  #  cv2.waitKey(25)
#    if AllCatsVideoWriter is None:
#       h, w, d = OutMain.shape
#       AllCatsVideoWriter = cv2.VideoWriter(OutVideoAll, fourcc, 20.0, (w, h))
#    AllCatsVideoWriter.write(OutMain)
#-----------------------------------------------------------------------------------------------------------------------------
print("Finished")
AllCatsVideoWriter.release()
MainCatsVideoWriter.release()
cap.release()







