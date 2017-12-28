# deepfakes-faceswap-GAN
Adding Adversarial loss and perceptual loss (VGGface) to deepfakes' auto-encoder architecture.

# Descriptions
* [FaceSwap_GAN_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_github.ipynb): This jupyter notebook does the following jobs:
  1. Build a GAN model. 
  2. Train the GAN from scratch. 
  3. Detect faces in an image using dlib's cnn model. 
  4. Use GAN to transform detected face into target face. 
  5. Use moviepy module to output a video clip with swapped face.  

* [dlib_video_face_detection.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/dlib_video_face_detection.ipynb): This jupyter notebook does the following jobs: 
  1. Detect/Crop faces in a video using dlib's cnn model. 
  2. Pack cropped face images into a zip file.
 
* Training data: 5k/9k face images for each target. Raw materials are crawled from pinterest and youtube, and then cropped using dlib. Face alignment was not applied on training data.

# Results

In below are results that show trained models transforming Hinako Sano ([佐野ひなこ](https://ja.wikipedia.org/wiki/%E4%BD%90%E9%87%8E%E3%81%B2%E3%81%AA%E3%81%93), left) to Emi Takei ([武井咲](https://ja.wikipedia.org/wiki/%E6%AD%A6%E4%BA%95%E5%92%B2), right).  

## 1. deepfakes' autorecoder

It should be mentoined that the result of autoencoder (AE) can be much better if we trained it for longer.

![AE GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/AE_sh_test.gif)  
 ![AE_results](https://github.com/shaoanlu/faceswap-GAN/raw/master/AE_results.png)

## 2. GAN (adding adversarial loss)

Adversarial loss improves resolution of generated images. We can see the differences on eyes, mouth and teeth compare to AE. Furthermore, I applied a smoothing mask on generated images before pasting it back to original face, thus the result looks more natrual.

![GAN_GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/woPL_sh_test3.gif)
![GAN_results](https://github.com/shaoanlu/faceswap-GAN/raw/master/woPL_results.png)

## 3. GAN (adding adversarial loss and [VGGFace](https://github.com/rcmalli/keras-vggface) perceptual loss)

When perceptual loss is apllied, the movemnet of eyeballs becomes more realistic and cnosistent with input face. However, training time is doubled ~ tripled.

![GAN_PL_GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/PL_sh_test3.gif)
![GAN_PL_results](https://github.com/shaoanlu/faceswap-GAN/raw/master/wPL_results.png)

###### Source video: [佐野ひなことすごくどうでもいい話？(遊戯王)](https://www.youtube.com/watch?v=tzlD1CQvkwU)

The following figure shows output of mdoel trained with/wihtout perceptual loss (PL). Note that the input image is not in the training data, thus the outputs are blurry than above images. This somehow indicates limitation of my faceswap model.
![Comp PL](https://github.com/shaoanlu/faceswap-GAN/raw/master/comparison_PL.PNG)

## Notes:
1. BatchNorm/InstanceNorm: Caused input/output skin color inconsistency when the 2 training dataset had different skin color dsitribution (light condition, shadow, etc.).
2. Increasing perceptual loss weighting factor (to 1) unstablized training. But the weihgting [.01, .1, .1] I used is not optimal either.
3. In the encoder architecture, flattening Conv2D and shrinking it to Dense(1024) is crutial for model to learn semantic features, or face representation. If we used Conv layers only (which means larger dimension), will it learn features like visaul descriptors? ([source paper](https://arxiv.org/abs/1706.02932v2), last paragraph of sec 3.1)
4. Transform Emi Takei to Hinko Sano gave suboptimal results, due to imbalanced training data that over 65% of images of Hinako Sano came from the same video series.
5. Mixup technique ([arXiv](https://arxiv.org/abs/1710.09412)) and least squares loss function are adopted ([arXiv](https://arxiv.org/abs/1712.06391)) for training GAN. However, I did not do any ablation experiment on them. Don't know how much impact they had on outputs.

# Requirements

* keras 2
* Tensorflow 1.3 
* Python 3
* OpenCV
* dlib
* face_recognition
* moviepy

## Acknowledgments
Code borrows from [tjwei](https://github.com/tjwei/GANotebooks) and [deepfakes](https://github.com/deepfakes/faceswap). The generative network is adopted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
