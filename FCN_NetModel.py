
import scipy.misc as misc
import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# FCN Net model class for semantic segmentation
##############################################This is a standart FCN with the lat layer split into prediction of binary map for every class########################################################################333
class Net(nn.Module):
######################################Build FCN net layer##################################################################################
    def __init__(self, CatDict):
        # Generate standart FCN PSP net for image segmentation with only image as input
        # --------------Build layers for standart FCN with only image as input------------------------------------------------------
            super(Net, self).__init__()
            # ---------------Load pretrained  Resnet 50 encoder----------------------------------------------------------
            self.Encoder = models.resnet101(pretrained=True)
            # ---------------Create Pyramid Scene Parsing PSP layer -------------------------------------------------------------------------
            self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8]

            self.PSPLayers = nn.ModuleList()  # [] # Layers for decoder
            for Ps in self.PSPScales:
                self.PSPLayers.append(nn.Sequential(
                    nn.Conv2d(2048, 1024, stride=1, kernel_size=3, padding=1, bias=True)))
                # nn.BatchNorm2d(1024)))
            self.PSPSqueeze = nn.Sequential(
                nn.Conv2d(4096, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
            # ------------------Skip conncetion layers for upsampling-----------------------------------------------------------------------------
            self.SkipConnections = nn.ModuleList()
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(256, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            # ------------------Skip squeeze applied to the (concat of upsample+skip conncection layers)-----------------------------------------------------------------------------
            self.SqueezeUpsample = nn.ModuleList()
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 256, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))


            # ----------------Final prediction layers------------------------------------------------------------------------------------------
            self.OutLayersList =nn.ModuleList()
            self.OutLayersDict={}
            for f,nm in enumerate(CatDict):

                    self.OutLayersDict[nm]= nn.Conv2d(256, 2, stride=1, kernel_size=3, padding=1, bias=False)
                    self.OutLayersList.append(self.OutLayersDict[nm])

##########################################Run inference################################################################################################################
    def forward(self,Images,UseGPU=True,TrainMode=True, FreezeBatchNormStatistics=False):

#----------------------Convert image to pytorch and normalize values-----------------------------------------------------------------
                RGBMean = [123.68,116.779,103.939]
                RGBStd = [65,65,65]
                if TrainMode:
                        tp=torch.FloatTensor
                else:
                    self.half()
                    tp=torch.HalfTensor
                   # self.eval()
                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(float)), requires_grad=False).transpose(2,3).transpose(1, 2).type(tp)
                if FreezeBatchNormStatistics==True: self.eval()
#---------------Convert to cuda gpu or CPU -------------------------------------------------------------------------------------------------------------------
                if UseGPU:
                    InpImages=InpImages.cuda()
                    self.cuda()
                else:
                    self=self.cpu()
                    self.float()
                    InpImages=InpImages.type(torch.float).cpu()
#----------------Normalize image values-----------------------------------------------------------------------------------------------------------
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # normalize image values
                x=InpImages
#--------------------------------------------------------------------------------------------------------------------------
                SkipConFeatures=[] # Store features map of layers used for skip connection
#---------------Run Encoder first layer-----------------------------------------------------------------------------------------------------
                x = self.Encoder.conv1(x)
                x = self.Encoder.bn1(x)
#-------------------------Run remaining encoder layer------------------------------------------------------------------------------------------
                x = self.Encoder.relu(x)
                x = self.Encoder.maxpool(x)
                x = self.Encoder.layer1(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer2(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer3(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer4(x)
#------------------Run psp  Layers----------------------------------------------------------------------------------------------
                PSPSize=(x.shape[2],x.shape[3]) # Size of the original features map

                PSPFeatures=[] # Results of various of scaled procceessing
                for i,PSPLayer in enumerate(self.PSPLayers): # run PSP layers scale features map to various of sizes apply convolution and concat the results
                      NewSize=(np.array(PSPSize)*self.PSPScales[i]).astype(np.int)
                      if NewSize[0] < 1: NewSize[0] = 1
                      if NewSize[1] < 1: NewSize[1] = 1

                      # print(str(i)+")"+str(NewSize))
                      y = nn.functional.interpolate(x, tuple(NewSize), mode='bilinear')
                      #print(y.shape)
                      y = PSPLayer(y)
                      y = nn.functional.interpolate(y, PSPSize, mode='bilinear')

                #      if np.min(PSPSize*self.ScaleRates[i])<0.4: y*=0
                      PSPFeatures.append(y)
                x=torch.cat(PSPFeatures,dim=1)
                x=self.PSPSqueeze(x)
#----------------------------Upsample features map  and combine with layers from encoder using skip  connection-----------------------------------------------------------------------------------------------------------
                for i in range(len(self.SkipConnections)):
                  sp=(SkipConFeatures[-1-i].shape[2],SkipConFeatures[-1-i].shape[3])
                  x=nn.functional.interpolate(x,size=sp,mode='bilinear') #Resize
                  x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1-i]),x), dim=1)
                  x = self.SqueezeUpsample[i](x)
                  # print([i])
                  # print(self.SqueezeUpsample[i][0].weight.sum())
#---------------------------------Final prediction-------------------------------------------------------------------------------
                self.OutProbDict = {}
                self.OutLbDict = {}
             #   print("=====================================================")
#===============Run prediction for each class as binary  mask========================================================================================
                for nm in self.OutLayersDict:
                    # print(nm)
                    # print((self.OutLayersDict[nm].weight.mean().cpu().detach().numpy()))

                    l=self.OutLayersDict[nm](x)
                 #   l = self.OutLayersDict[nm](x) # Make prediction per pixel
                    l = nn.functional.interpolate(l,size=InpImages.shape[2:4],mode='bilinear') # Resize to original image size
                    Prob = F.softmax(l, dim=1)  # Calculate class probability per pixel
                    tt, Labels = l.max(1)  # Find label per pixel
                    self.OutProbDict[nm]=Prob
                    self.OutLbDict[nm] = Labels
#********************************************************************************************************
                return  self.OutProbDict,self.OutLbDict








