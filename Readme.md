# Detecting and segmenting and classifying materials inside vessels in images using a fully convolutional neural net.


Neural net that given an image, detects and segments and classifies the vessels (mainly transparent vessels) and the materials inside the vessels in the image (Figure 1). The net marks the vessel region, the filled region inside the vessel, and the specific region of various a phase of materials such as liquid, solid, foam, suspension, powder, granular... In addition, the net also predicts the region of the vessel labels cork and other parts (such as valves in separatory funnels).  


This code with a trained model that can be run of the box without training can be download from (here) [].
## General 
The net focus on detecting vessels and their content materials in images. The focus is on both chemistry lab setting and general everyday setting (beverage, kitchen..) but should work in any conditions or setting. The net should recognize  any transparent vessel (bottle/glass /or lab vessel) and their content and some none transparent vessels in any general enviroment and setting. The accuracy of the net is relatively high in detecting and classifying vessels, filled regions, liquid regions, and solid regions. The classification accuracy  fine grained material classses such as foams, powder, gels, etc.. is  lower. If you encounter cases on which the net perform badly, please send me the images so I can use them to improve the net.


 ![](/Figure1.jpg)
Figure 1) Input images and output results of the net.
 


## Details input/output
The input for the net is a standard image (Figure 1 right).
The net output of the region of the vessel/fill level and other materials phases, and vessel parts in the image (Figure 1 left). For each class, the net will output a mask with the region of the image corresponding to this class in the image.








# Requirements
## Hardware
For using the [trained net](), no specific hardware is needed, but the net will run much faster on Nvidia GPU.

For training the net an Nvidia GPU is needed (the net was trained on Titan XT, and also on RTX 2070 with similar results)
## Software:
This network was run with Python 3.7 [Anaconda](https://www.anaconda.com/download/) with  [Pytorch 1](https://pytorch.org/) and OpenCV packages.




# Setup for running prediction
1) Install [Anaconda](https://www.anaconda.com/download/)
2) Install [Pytorch](https://pytorch.org/)
2) Install OpenCV
3) Download the code with trained model weight from [Here]


# Tutorial


# Running inference on image and predicting segment mask
1. Download the code with trained model weight from [Here]() or train the model yourself using the instructions of the Training section.
2. Open the RunPredictionOnFolder.py script.
3. Set the path to the folder where the images are stored to the: InputDir parameter (all the images in the input folder should be in .jpg or .png format)
4. Set the output folder where the output will be stored to the: OutDir Parameter.
5. Run script. 
6. Output: predicted region for each input image and class would appear in the OutDir folder. 
Note: RunPredictionOnFolder.py should run out of the box as is using the sample images provided.
  ## Additional parameters:
* If you train the net yourself, set the path to your  trained model  in the Trained_model_path parameter
*  If you have a Nvidia GPU and Cuda installed, set the UseGPU parameter to True (this will allow the net to achieve a much faster running time).
* Changing FreezeBatchNormStatistics parameter from False to True might change the segmentation quality for better or worst (and so does changing the image size)




# Training
1. Download the LabPics data set from [Here](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) or [here](https://drive.google.com/file/d/1gfaM_6eZjtg7dkFShGl1gIfsXzj1KjIX/view?usp=sharing)
2. Open the Train.py script
3. Set the path to the LabPics to the TrainFolderPath parameters.
4. Run the script 
5. Output trained model will appear in the /log subfolder or any folder set in Trained model Path.


* Around 28000 training steps which correspond to two days training with Titan TX should give a model of the same performance as the one that can be download from [here]
# Code file structure
RunPredictionOnFolder.py Run prediction on image using pre-trained image
Train.py: Training the net of the [LabPics](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) dataset
ChemReader.py: File reader for the LabPics dataset (used by the Train.py script)
FCN_NetModel.py: The class containing the neural net module.
Evaluator.py: Evaluate the net performance during training (Used by Train.py)
CategoryDictionary.py: List of classes and subclasses used by the net and LabPics dataset.
Logs folder: Folder where the trained model and train relating data is stored
InputImages Folder: Example input images for the net


# Notes/Thanks
The training data for the [LabPics](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) dataset and images for this path were taken from Youtube channels such as NileRide, NurdeRage, and Chemplayer, douglas lab, Koen2All and from Instagram channels such as Chemlife organic Chemistry lab,Chemistry and me, Ministry Of Chemistry , vacuum distillation.
Work was done in the Matter Lab (under Alan Aspuru Gusnik group) Toronto.
The LabPics dataset was made by Mor Bismuth.

# Links
[LabPics dataset for annotated images of liquid, solid and foam materials in mostly transperent vessels in Lab setting and general everyday setting ]((https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) [or here](https://drive.google.com/file/d/1gfaM_6eZjtg7dkFShGl1gIfsXzj1KjIX/view?usp=sharing)
[Same Code with train model with train model]
