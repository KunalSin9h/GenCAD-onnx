# GenCAD: Image-conditioned Computer-Aided Design Generation with Transformer-based Contrastive Representation and Diffusion Priors

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A1234.56789-b31b1b.svg)](https://arxiv.org/abs/2409.16294)
[![Website](https://img.shields.io/badge/Project%20Page-Link-blue)](https://gencad.github.io/)

---

## Dataset 

The dataset for GenCAD can be downloaded from [here](https://drive.google.com/drive/folders/1M0dPr5kILGY9HTRCHox1vLLDhhxJWl_C?usp=sharing) and place it in the data directory. 

## Pretrained models

Pretrained models can be found [here](https://drive.google.com/drive/folders/1Ej7wdtlqT5P-SoUf3gsZXD8b78XqhiI5?usp=sharing) and should be placed in the `data/ckpt` directory. 

## Training the CSR model 

Run the following command to train the CSR model  from scratch. If you want to start with a checkpoint just add the `-ckpt` flag with the path, e.g. `-ckpt "model/ckpt/ae_ckpt_epoch1000.pth"`. 

 ```python train_gencad.py csr -name test -gpu 0```

## Training the CCIP model 

Run the following command to train the CSR model  from scratch. If you want to start with a checkpoint just add the `-ckpt` flag with the path, e.g. `-ckpt "model/ckpt/ae_ckpt_epoch1000.pth"`. Note that you must provide the pretrained cad autoencoder (csr model) checkpoint which is kept frozen during the image encoder training. 

 ```python train_gencad.py ccip -name test -gpu 0 -cad_ckpt "model/ckpt/ae_ckpt_epoch1000.pth"```


## Training the Diffusion Prior model 

Run the following command to train the DP model from scratch. Note that you must provide the image embeddings and cad embeddings to train the DP model. These embeddings are obtained by passing the entire training dataset through the pretrained image encoder and cad encoder respectively. 

``` python train_gencad.py dp -name test -gpu 0 -cad_emb 'data/embeddings/cad_embeddings.h5' -img_emb 'data/embeddings/sketch_embeddings.h5'```

## Inference 

Running the inference code will generate CAD and you can save the CAD as STL or STEP or image. To get images in a headless way, please use `xvfb-run` infront of the main code. The inference code is very straightforward and easy to modify. 

```xvfb-run python inference_gencad.py```

## Evaluation 

will be updated soon. 

## Visualization 

We provide a simple script to visualize any STL file using OPENCASCADE and save the image in `.png` format. Just run the following code or modify as you want. 

```python stl2img.py -src path/to/stl/files -dst path/to/save/images```


