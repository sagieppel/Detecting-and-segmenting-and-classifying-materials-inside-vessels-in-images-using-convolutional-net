##########################Train fully convolutional net on the labpics Dataset######################################################################################################
#...............................Imports..................................................................

import os
import torch
import numpy as np
import ChemReader
import FCN_NetModel as FCN # The net Class
import CategoryDictionary as CatDic
import Evaluator
import scipy.misc as misc
#-------------------------------------Input parameters-----------------------------------------------------------------------
TrainFolderPath=r"/scratch/gobi2/seppel/Chemscape/LabPicsV1/"

ChemTrainDir=TrainFolderPath+r"/Complex/Train//" #Input training data from the LabPics dataset
ChemTestDir=TrainFolderPath+r"/Complex/Test//" # Input testing data  from the LabPics dataset


#----------------------------------------------------------------------------------------------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
if not os.path.exists(TrainedModelWeightDir): os.mkdir(TrainedModelWeightDir)
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""



#-----------------------------------------Input parameters---------------------------------------------------------------------
Learning_Rate_Init=1e-5 # Initial learning rate
Learning_Rate=1e-5 # learning rate
#Learning_Rate_Decay=Learning_Rate[0]/40 # Used for standart
Learning_Rate_Decay=Learning_Rate/20
StartLRDecayAfterSteps=100000
MaxBatchSize=7 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width
MaxPixels=340000*3# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(10000000010) # Max  number of training iteration
InitStep=0
#-----------------Generate evaluator class for net evaluating------------------------------------------------------------------------------------------------------------------------------------------------

Eval=Evaluator.Evaluator(ChemTestDir,TrainedModelWeightDir+"/Evaluat.xls")

#----------------------------------------Create reader for /labpics data set--------------------------------------------------------------------------------------------------------------
ChemReader=ChemReader.Reader(MainDir=ChemTrainDir,MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,TrainingMode=True)
#=========================Load Paramters====================================================================================================================
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"): Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"): Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate_Init.npy"): Learning_Rate_Init=np.load(TrainedModelWeightDir+"/Learning_Rate_Init.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=FCN.Net(CatDic.CatNum) # Create net and load pretrained encoder path
if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer
#--------------------------- Create list for saving statistics----------------------------------------------------------------------------------------------------------
AVGLoss={}
for nm in CatDic.CatLossWeight:
    AVGLoss[nm]=-1

AVGtotalLoss=-1
#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
txt="Iteration\t Learning Rate\t Learning rate\t"
for nm in AVGLoss: txt+="\t"+nm+" loss"
f.write(txt+"\n")
f.close()
#..............Start Training loop: Main Training....................................................................

print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop
    Imgs, Ignore, AnnMaps, AnnMapsBG = ChemReader.LoadBatch()

   #  for oo in range(PredMask.shape[0]):
   #     # misc.imshow(Imgs[oo])
   #    #  Imgs[oo,:,:,0] *=1 - PredMask[oo,:,:]
   #      im= Imgs[oo].copy()
   #      im[:,:,0] *= 1 - GTMask[oo,:,:]
   #      im[:, :, 1] *= 1 - PredMask[oo,:,:]
   #      print(IOU[oo])
   # #     misc.imshow((PredMask[oo,:,:]*0+GTMask[oo,:,:]))
   #      misc.imshow(np.concatenate([Imgs[oo],im],axis=0))


    OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=True) # Run net inference and get prediction
    Net.zero_grad()
#------------------------Calculate Loss for each class and sum the losses---------------------------------------------------------------------------------------------------
    Loss = 0
    LossByCat={}
    ROI = torch.autograd.Variable(torch.from_numpy((1-Ignore).astype(np.float32)).cuda(), requires_grad=False)
    for nm in OutProbDict:
        if CatDic.CatLossWeight[nm]<=0: continue
        if nm in AnnMaps:
            GT=torch.autograd.Variable( torch.from_numpy(AnnMaps[nm].astype(np.float32)).cuda(), requires_grad=False)
            LossByCat[nm]=-torch.mean(ROI*(GT * torch.log(OutProbDict[nm][:,1,:,:] + 0.0000001)+(1-GT) * torch.log(OutProbDict[nm][:,0,:,:] + 0.0000001)))
            Loss=LossByCat[nm]*CatDic.CatLossWeight[nm]+Loss



    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight

#-----------Update loss statitics-------------------------------------------------------------------------------------------------
    if AVGtotalLoss == -1:
        AVGtotalLoss = float(Loss.data.cpu().numpy())  # Calculate average loss for display
    else:
        AVGtotalLoss = AVGtotalLoss * 0.999 + 0.001 * float(Loss.data.cpu().numpy())

    for nm in LossByCat:
        if AVGLoss[nm]==-1:  AVGLoss[nm]=float(LossByCat[nm].data.cpu().numpy()) #Calculate average loss for display
        else: AVGLoss[nm]= AVGLoss[nm]*0.999+0.001*float(LossByCat[nm].data.cpu().numpy()) # Intiate runing average loss
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 2000 == 0 and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/Learning_Rate_Init.npy",Learning_Rate_Init)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 10000 == 0 and itr>1: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#--------------------------Evaluate trained net-------------------------------------------------------------------------
    if itr % 10000 == 0:
        Eval.Eval(Net,itr)
#......................Write and display train loss..........................................................................
    if itr % 50==0: # Display train loss

        txt="\nIteration\t="+str(itr)+"\tLearning Rate\t"+str(Learning_Rate)+"\tInit_LR=\t"+str(Learning_Rate_Init)+"\tLoss=\t"+str(AVGtotalLoss)+"\t"
        for nm in AVGLoss:
            txt+="\t"+nm+"=\t"+str(AVGLoss[nm])
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
#----------------Update learning rate fractal manner-------------------------------------------------------------------------------
    if itr%10000==0 and itr>=StartLRDecayAfterSteps:
        Learning_Rate-= Learning_Rate_Decay
        if Learning_Rate<=1e-7:
            Learning_Rate_Init-=2e-6
            if Learning_Rate_Init<1e-6: Learning_Rate_Init=1e-6
            Learning_Rate=Learning_Rate_Init*1.00001
            Learning_Rate_Decay=Learning_Rate/20
        print("Learning Rate="+str(Learning_Rate)+"   Learning_Rate_Init="+str(Learning_Rate_Init))
        print("======================================================================================================================")
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer
        torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
