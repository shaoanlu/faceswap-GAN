## Notebooks that are not maintained anymore are in this folder.

### faceswap-GAN v2.1
* [FaceSwap_GAN_v2.1_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/legacy/FaceSwap_GAN_v2.1_train.ipynb)
  - A experimental model that provides architectures like VAE and [XGAN](https://arxiv.org/abs/1711.05139).
  - In video conversion, it ultilizes FCN for face segmentation to generate a hybrid alpha mask.

V2.1 model is an improved architecture in order to stablize training. The architecture is greatly inspired by [XGAN](https://arxiv.org/abs/1711.05139) ~~and [MS-D neural network](http://www.pnas.org/content/115/2/254)~~. (Note: V2.1 script is experimental and not well-maintained)
  - V2.1 model  provides three base architectures: (i) XGAN, (ii) VAE-GAN, and (iii) a variant of v2 GAN. (default `base_model="GAN"`)
  - FCN8s for face segmentation is introduced to improve masking in video conversion (default `use_FCN_mask = True`).
    - To enable this feature, keras weights file should be generated through jupyter notebook provided in [this repo](https://github.com/shaoanlu/face_segmentation_keras).

### faceswap-GAN v2
* [FaceSwap_GAN_v2_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/legacy/FaceSwap_GAN_v2_train.ipynb)
  - Notebook for training the version 2 GAN model.
  - Video conversion functions are also included.
  
* [FaceSwap_GAN_v2_test_video_MTCNN.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/legacy/FaceSwap_GAN_v2_test_video_MTCNN.ipynb)
  - Notebook for generating videos. Use MTCNN for face detection.

* [faceswap_WGAN-GP_keras_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/lefacy/faceswap_WGAN-GP_keras_github.ipynb)
  - This notebook is an independent training script for a GAN model of [WGAN-GP](https://arxiv.org/abs/1704.00028) in which perceptual loss is discarded for simplicity. 
  - Training can be start easily as the following:
  ```python
  gan = FaceSwapGAN() # instantiate the class
  gan.train(max_iters=10e4, save_interval=500) # start training
  ```
* [FaceSwap_GAN_v2_sz128_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynb)
  - This notebook is an independent script for a model with 128x128 input/output resolution.
  
### faceswap-GAN v1
* [FaceSwap_GAN_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/legacy/FaceSwap_GAN_github.ipynb)
  - V1 model directly predicts color output images without masking.
  - Video conversion functions are also included.
