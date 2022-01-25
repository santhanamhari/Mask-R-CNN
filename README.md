# Mask-R-CNN

We implemented Faster R-CNN structure now repurposed to accomodate a Feature Pyramid Network and extend it by adding a mask branch to the network head architecture. We also use RoIAlign to correct misalignments due to pooling. The mask head takes in RoI aligned features from the Box Head network and uses a FCN to predict the masks. We employed a split 3 stage strategy in training to alleviate the cumbersome nature of training the entire pipeline together. Hence, the RPN, Box Head, and Mask Head were trained separately with the previous networks weights frozen for the subsequent training. The RPN had a point wise pixel accuracy of 0.65 and the Box Head returned an MAP of 0.62 with an AP of 0.63 for the 'vehicle' class, 0.58 for the 'person' class and 0.64 for the 'animal' class. We also achieve strong final masks produced from MaskHead, with multiple instances across many scales accurately detected. \href{https://drive.google.com/file/d/1-CXbW6fTA_IXyao3wVjcT5uRXfSbAK0j/view?usp=sharing}{VIDEO}

## RPN

<img width="299" alt="rpn1" src="https://user-images.githubusercontent.com/40223805/150955449-4312235f-db6b-4f07-8cb4-bd07bf8426a9.png">

<img width="278" alt="rpn2" src="https://user-images.githubusercontent.com/40223805/150955485-874696c8-f4ea-451f-b644-bff2acbd1a38.png">

<img width="270" alt="rpn3" src="https://user-images.githubusercontent.com/40223805/150955503-761382b9-7c9f-41c5-bb9e-8ff14a9c5be4.png">

## BoxHead

<img width="523" alt="boxhead1" src="https://user-images.githubusercontent.com/40223805/150955554-896aa03f-6cce-485b-97de-6513764aabf2.png">

<img width="536" alt="boxhead2" src="https://user-images.githubusercontent.com/40223805/150955616-e2b09962-c33a-4f3f-8786-71fb837ef775.png">

<img width="538" alt="boxhead3" src="https://user-images.githubusercontent.com/40223805/150955632-6ba3a384-622f-4157-8012-088fbc4c5557.png">


## MaskHead


<img width="236" alt="maskhead3" src="https://user-images.githubusercontent.com/40223805/150955670-b141128d-525e-4ae7-bded-1dda44de79d2.png">

<img width="256" alt="maskhead1" src="https://user-images.githubusercontent.com/40223805/150955691-6ab8d9b8-943c-4ae3-b221-e9aa19acaf40.png">

<img width="269" alt="maskhead4" src="https://user-images.githubusercontent.com/40223805/150955706-d682beb0-0130-46aa-8d12-d6485e07b3f6.png">

<img width="391" alt="maskhead8" src="https://user-images.githubusercontent.com/40223805/150955810-e0508617-0968-4c9a-97e9-bbef51f2aa1b.png">

<img width="250" alt="maskhead5" src="https://user-images.githubusercontent.com/40223805/150955838-1a8d193d-9bce-4262-a72b-5bfa99db4901.png">
