# faceswap-GAN
Adding Adversarial loss and perceptual loss (VGGface) to deepfakes' auto-encoder architecture.

## Descriptions
### GAN-v1
* [FaceSwap_GAN_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_github.ipynb)

  1. Build a GAN model. 
  2. Train the GAN from scratch. 
  3. Use GAN to swap a single face image to target face.
  4. Detect faces in an image using dlib's cnn model. 
  5. Use GAN to transform detected face into target face. 
  6. Use moviepy module to output a video clip with swapped face.  
  
### GAN-v2
* [FaceSwap_GAN_v2_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_train.ipynb): Detailed training procedures can be found in this notebook.
  1. Build a GAN model. 
  2. Train the GAN from scratch. 
  3. (Optoinal) Detect faces in an image using dlib's cnn model. 
  4. (Optoinal) Use GAN to transform detected face into target face. 
  5. (Optoinal) Use moviepy module to output a video clip with swapped face.
  
* [FaceSwap_GAN_v2_test_img.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_img.ipynb): Provides `swap_face()` function that require less VRAM.
  1. Load trained model.
  2. Swap a single face image to target face.
  
* [FaceSwap_GAN_v2_test_video.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video.ipynb)
  1. Load trained model.
  2. Detect faces in an image using dlib's cnn model. 
  3. Use GAN to transform detected face into target face. 
  4. Use moviepy module to output a video clip with swapped face.    
  
### Others
* [dlib_video_face_detection.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/dlib_video_face_detection.ipynb)
  1. Detect/Crop faces in a video using dlib's cnn model. 
  2. Pack cropped face images into a zip file.
 
* Training data: Face images are supposed to be in `./faceA/` and `./faceB/` folder for each target respectively. Face images can be of any size. (Updated 3, Jan., 2018)

## Results

In below are results that show trained models transforming Hinako Sano ([佐野ひなこ](https://ja.wikipedia.org/wiki/%E4%BD%90%E9%87%8E%E3%81%B2%E3%81%AA%E3%81%93), left) to Emi Takei ([武井咲](https://ja.wikipedia.org/wiki/%E6%AD%A6%E4%BA%95%E5%92%B2), right).  
###### Source video: [佐野ひなことすごくどうでもいい話？(遊戯王)](https://www.youtube.com/watch?v=tzlD1CQvkwU)
### 1. Autorecoder

Autoencoder based on deepfakes' script. It should be mentoined that the result of autoencoder (AE) can be much better if we trained it for longer.

![AE GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/AE_sh_test.gif)![AE_results](https://github.com/shaoanlu/faceswap-GAN/raw/master/readme_imgs/AE_results.png)

### 2. [Generative Adversarial Network](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/), GAN (version 1)

**Improved output resolution:** Adversarial loss improves resolution of generated images. In addition, when perceptual loss is apllied, the movemnet of eyeballs becomes more realistic and consistent with input face.

![GAN_PL_GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/smoothedBboxPL_sh_test3.gif)![GAN_PL_results](https://github.com/shaoanlu/faceswap-GAN/raw/master/readme_imgs/wPL_results_resized.png)

**VGGFace[(GitHub repo)](https://github.com/rcmalli/keras-vggface) perceptual loss (PL):** The following figure shows nuanced eyeballs direction in model output trained with/wihtout PL. 

![Comp PL](https://github.com/shaoanlu/faceswap-GAN/raw/master/readme_imgs/comparison_PL_rev.png)

**Smoothed bounding box (Smoothed bbox):** Exponential moving average of bounding box position over frames is introduced to eliminate jittering on the swapped face. See the below gif for comparison.

![bbox](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/bbox_comp_annotated.gif)
  - A. Source face.
  - B. Swapped face, using smoothing mask (smooths the edges of output image when pasting it back to input image).
  - C. Swapped face, using smoothing mask and face alignment.
  - D. Swapped face, using smoothing mask and smoothed bounding box.

### 3. [Generative Adversarial Network](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/), GAN (version 2)

**Version 1 features:** Most of features in version 1 are inherited, including perceptual loss and smoothed bbox.

**Segmentation mask prediction:** Model learns a proper mask that helps on handling occlusion, eliminating artifacts on bbox edges, and producing natrual skin tone.

![mask0](https://github.com/shaoanlu/faceswap-GAN/raw/master/readme_imgs/comp_mask_rev.png)

![mask1](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/mask_comp1.gif)  ![mask2](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/mask_comp2.gif)
  - Left: Source face.
  - Middle: Swapped face, before masking.
  - Right: Swapped face, after masking.

The following gif presents a comparison of version 1 and version 2 output result.

![ver_comp_gif](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/ver1ver2_bboxedge.gif)
  - Left: GAN version 1. A barely noticeable bounding box is shown (see the bangs and hair color around cheeks).
  - Right: GAN version 2. Version 2 eliminates bbox artifacts and produces natrual skin tone.

**Mask visualization**: The following gif shows output mask & face bounding box.

![mask_vis](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/mask_vis_rev.gif)
  - Left: Source face.
  - Middle: Swapped face, after masking.
  - Right: Mask heatmap & face bounding box.

## Requirements

* keras 2
* Tensorflow 1.3 
* Python 3
* OpenCV
* dlib
* [face_recognition](https://github.com/ageitgey/face_recognition)
* [moviepy](http://zulko.github.io/moviepy/)

## Notes:
1. BatchNorm/InstanceNorm: Caused input/output skin color inconsistency when the 2 training dataset had different skin color dsitribution (light condition, shadow, etc.).
2. Increasing perceptual loss weighting factor (to 1) unstablized training. But the weihgting [.01, .1, .1] I used is not optimal either.
3. In the encoder architecture, flattening Conv2D and shrinking it to Dense(1024) is crutial for model to learn semantic features, or face representation. If we used Conv layers only (which means larger dimension), will it learn features like visaul descriptors? ([source paper](https://arxiv.org/abs/1706.02932v2), last paragraph of sec 3.1)
4. Transform Emi Takei to Hinko Sano gave suboptimal results, due to imbalanced training data that over 65% of images of Hinako Sano came from the same video series.
5. Mixup technique ([arXiv](https://arxiv.org/abs/1710.09412)) and least squares loss function are adopted ([arXiv](https://arxiv.org/abs/1712.06391)) for training GAN. However, I did not do any ablation experiment on them. Don't know how much impact they had on outputs.

## Acknowledgments
Code borrows from [tjwei](https://github.com/tjwei/GANotebooks) and [deepfakes](https://pastebin.com/hYaLNg1T). The generative network is adopted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
