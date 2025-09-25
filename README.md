# Multi stage attention U-net for fundus imaging segmentation
# Introduction
Fundus imaging segmentation plays a key role during ophthalmic diseases diagnosis. While manually segmentation is a time-consuming task and demands a specialist, in the last years an undefined number of works have been developed to automatically identify ocular structures and speed this task.

Here I propose a multi-stage and multi-task attention U-net for vessel, optic disc and optic disc cup segmentation in fundus images. It a 3-train stage model, each with a different loss function (BCE, Tversky and FocalTversky). This approach aims progressive learning and better generalization to improve the segmentation quality. 

# How to use:
- After the download set the work directory to the project folder;
- Set the dataset_images and dataset_masks with the images and masks;
  - Images and masks should be same name and format (accepted formats are: .png, .jpg, .jpeg and .tif)
- Set the number of epochs and patience (for early stopping)

# Results:
The model was tested in the following well stablished datasets: 
