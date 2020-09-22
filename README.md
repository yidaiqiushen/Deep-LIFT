# Deep-LIFT
A Pytorch implementation of the paper "Deep-LIFT: Deep Label Specific Feature Learning for Image Annotation", IEEE Transactions on Cybernetics (IEEE T-CYB), 2020.

## Requirements
- torch-1.3  
- Python 3.6  
- torchvision-0.2.0  
- torchnet  
- numpy

## Download pretrain models
checkpoint/coco
checkpoint/voc

## Testing
you can simply test the pretrain model in the following way:

       `python3 voc_gcn.py data/voc -e --resume checkpoint/voc2007/model_best.pth.tar`

## Training
you can adjust some hyperparameters and train the model in the following way:

       `python3 voc_gcn.py data/voc`

## Citation
If you find that Deep-LIFT helps your research, please cite our paper.

## Questions
For any additional questions, feel free to email lijunbing@tju.edu.cn.





