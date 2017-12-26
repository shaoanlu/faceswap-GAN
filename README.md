# deepfakes-faceswap-GAN
Adding Adversarial loss and perceptual loss (VGGface) to deepfakes' auto-encoder architecture.

# Descriptions
* [FaceSwap_GAN_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_github.ipynb): In this noetbook, we will: 
  1. Build a GAN model. 
  2. Train the GAN from scratch. 
  3. Detect faces in an image using dlib's cnn model. 
  4. Use GAN to transform detected face into target face. 
  5. Use moviepy module to output a video clip with swapped face.

* [dlib_video_face_detection.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/dlib_video_face_detection.ipynb): In this notebook. we will:
  1. Detect/Crop faces in video using dlib's cnn model. 
  2. Pack cropped face images into a zip file.

# Results

In below are results that shows trained models transforming Hinako Sano [(佐野ひなこ)](https://ja.wikipedia.org/wiki/%E4%BD%90%E9%87%8E%E3%81%B2%E3%81%AA%E3%81%93) to Emi Takei [(武井咲)](https://ja.wikipedia.org/wiki/%E6%AD%A6%E4%BA%95%E5%92%B2).  

## 1. deepfakes' autorecoder

It should be mentoined that the result of autoencoder (AE) can be much better if we trained it longer.

![AE GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/AE_sh_test.gif)

## 2. GAN (adding adversarial loss)

Adversarial loss improves resolution of generated images. We can see the differences on eyes, mouth and teeth compare to AE. Furthermore, I applied a smoothing mask on generated images before pasting it back to original face, thus the result looks more natrual.

![GAN_GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/woPL_sh_test3.gif)
![GAN_results](https://github.com/shaoanlu/faceswap-GAN/raw/master/woPL_results.png)

## 3. GAN (adding adversarial loss and [VGGFace](https://github.com/rcmalli/keras-vggface) perceptual loss)

When perceptual loss is apllied, the movemnet of eyeballs becomes more realistic (although hard to distinguish in the gifs). But the training time is doubled ~ tripled.

![GAN_PL_GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/PL_sh_test3.gif)
![GAN_PL_results](https://github.com/shaoanlu/faceswap-GAN/raw/master/wPL_results.png)

###### Source video: [佐野ひなことすごくどうでもいい話？(遊戯王)](https://www.youtube.com/watch?v=tzlD1CQvkwU)

# Requirements

* keras 2
* Tensorflow 1.3 
* Python 3
* dlib
* face_recodnition
* moviepy

## Acknowledgments
Code borrows from [tjwei](https://github.com/tjwei/GANotebooks) and [deepfakes](https://github.com/deepfakes/faceswap). The generative network is adopted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
