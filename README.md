# Image Clustering
Image clustering with ResNet-18 as encoder.


## Requirements

1. Python 3.6
2. torch & torchvision
3. tqdm
4. sklearn

## Methodology

1. Pretrain ResNet-18 on ImageNet (done by using built-in model of PyTorch);
2. Fine-tune the model on architecture style training set; 
3. Use ResNet as image encoder to encode the Kaiping images;
4. Cluster the obtained image embeddings from step 3.

## Usage

Put the folder *Architecture_Style* in *data*. Put 开平碉楼id in *data/test*.

Then
    
    python main.py [--pretrained] [--clustering] [--loadmodel [model_path]]
    
For training:
    
    python main.py --pretrained --batch_size 100

***delete pic_329 in 1_1***

Refer to models/train.py for commandline options.

Clustering results will be in the folder *images*.