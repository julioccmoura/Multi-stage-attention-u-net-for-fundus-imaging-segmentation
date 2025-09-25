# Multi stage attention U-net for fundus imaging segmentation
# Introduction
Fundus imaging segmentation plays a key role during ophthalmic diseases diagnosis. While manually segmentation is a time-consuming task and demands a specialist, in the last years an undefined number of works have been developed to automatically identify ocular structures and speed this task.
Here I propose a multi-stage and multi-task attention U-net for vessel, optic disc and optic disc cup segmentation in fundus images. It a 3-train stage model, each with a different loss function (BCE, Tversky and FocalTversky). This approach aims progressive learning and better generalization to improve the segmentation quality. 
