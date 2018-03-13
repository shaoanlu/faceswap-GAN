# faceswap-GAN
Adding Adversarial loss and perceptual loss (VGGface) to deepfakes' auto-encoder architecture.

## Updates
| Date          | Update        |
| ------------- | ------------- | 
| 2018-03-13      | **Model architecture**: V2.1 model now provides 3 base architectures: (i) XGAN, (ii) VAE-GAN, and (iii) a variant of v2 GAN. See "4. Training Phase Configuration" in [v2.1 notebook](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.1_train.ipynb) for detail.| 
| 2018-03-03      | **Model architecture**: Add a [new notebook](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.1_train.ipynb) which contains an improved GAN architecture. The architecture is greatly inspired by [XGAN](https://arxiv.org/abs/1711.05139) and [MS-D neural network](http://www.pnas.org/content/115/2/254).| 
| 2018-02-13      | **Video conversion**: Add a new video procesisng script using **[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)** for face detection. Faster detection with configurable threshold value. No need of CUDA supported dlib. (New notebook: [v2_test_vodeo_MTCNN](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video_MTCNN.ipynb))| 
| 2018-02-10      | **Video conversion**: Add a optional (default `False`) histogram matching function for color correction into video conversion pipeline. Set `use_color_correction = True` to enable this feature. (Updated notebooks: [v2_sz128_train](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynb), [v2_train](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_train.ipynb), and [v2_test_video](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video.ipynb))| 

## Descriptions
### GAN-v1
* [FaceSwap_GAN_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_github.ipynb)

  - Script for training the version 1 GAN model.
  - Video conversion functions are also included. 
  
### GAN-v2
* [FaceSwap_GAN_v2_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_train.ipynb) **(recommneded for trainnig)**
  - Script for training the version 2 GAN model.
  - Video conversion functions are also included.
  
* [FaceSwap_GAN_v2_test_video.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video.ipynb)
  - Script for generating videos.
  - Using face_recognition module for face detection.
  
* [FaceSwap_GAN_v2_test_video_MTCNN.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video_MTCNN.ipynb) **(recommneded for video conversion)**
  - Script for generating videos.
  - Using MTCNN for face detection. Does not reqiure CUDA supported dlib.
  
* [faceswap_WGAN-GP_keras_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/temp/faceswap_WGAN-GP_keras_github.ipynb)
  - This notebook contains a class of GAN mdoel using [WGAN-GP](https://arxiv.org/abs/1704.00028). 
  - Perceptual loss is discarded for simplicity. 
  - The WGAN-GP model gave me similar result with LSGAN model after tantamount (~18k) generator updates.
  ```python
  gan = FaceSwapGAN() # instantiate the class
  gan.train(max_iters=10e4, save_interval=500) # start training
  ```
* [FaceSwap_GAN_v2_sz128_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynb)
  - Input and output images have larger shape `(128, 128, 3)`.
  - Minor updates on the architectures: 
    1. Add instance normalization to generators and discriminators.
    2. Add additional regressoin loss (mae loss) on 64x64 branch output.
  - Not compatible with `_test_video` and `_test_video_MTCNN` notebooks above.
  
### Miscellaneous
* [dlib_video_face_detection.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/dlib_video_face_detection.ipynb)
  1. Detect/Crop faces in a video using dlib's cnn model. 
  2. Pack cropped face images into a zip file.
 
* Training data: Face images are supposed to be in `./faceA/` and `./faceB/` folder for each target respectively. Face images can be of any size.

## Results

In below are results that show trained models transforming Hinako Sano ([佐野ひなこ](https://ja.wikipedia.org/wiki/%E4%BD%90%E9%87%8E%E3%81%B2%E3%81%AA%E3%81%93)) to Emi Takei ([武井咲](https://ja.wikipedia.org/wiki/%E6%AD%A6%E4%BA%95%E5%92%B2)).  
###### Source video: [佐野ひなことすごくどうでもいい話？(遊戯王)](https://www.youtube.com/watch?v=tzlD1CQvkwU)
### 1. Autorecoder baseline

Autoencoder based on deepfakes' script. It should be mentoined that the result of autoencoder (AE) can be much better if we train it longer.

- **Results:**

  ![AE_results](https://www.dropbox.com/s/n9xjzhlc4llbh96/AE_results.png?raw=1)

### 2. Generative Adversarial Network, GAN (version 1)

- **Improved output quality:** Adversarial loss improves reconstruction quality of generated images.

  ![GAN_PL_results](https://www.dropbox.com/s/ex7z8upst0toyf0/wPL_results_resized.png?raw=1).

- **[VGGFace](https://github.com/rcmalli/keras-vggface) perceptual loss:** Perceptual loss improves direction of eyeballs to be more realistic and consistent with input face.

- **Smoothed bounding box (Smoothed bbox):** Exponential moving average of bounding box position over frames is introduced to eliminate jitter on the swapped face. 

### 3. Generative Adversarial Network, GAN (version 2)

- **Version 1 features:** Most of the features in version 1 are inherited, including perceptual loss and smoothed bbox.

- **Unsupervised segmentation mask:** Model learns a proper mask that helps on handling occlusion, eliminating artifacts on bbox edges, and producing natrual skin tone.

  ![mask1](https://www.dropbox.com/s/do3gax2lmhck941/mask_comp1.gif?raw=1)  ![mask2](https://www.dropbox.com/s/gh0yq26qkr31yve/mask_comp2.gif?raw=1)
    - From left to right: source face, swapped face (before masking), swapped face (after masking).

  ![mask_vis](https://www.dropbox.com/s/q6dfllwh71vavcv/mask_vis_rev.gif?raw=1)
    - From left to right: source face, swapped face (after masking), mask heatmap.
  
- **Optional 128x128 input/output resolution**: Increase input and output size from 64x64 to 128x128.

- **Mask refinement**: VGGFace ResNet50 is introduced for mask refinement (as the preceptual loss). The following figure shows generated masks before/after refinement. Input faces are from [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

  ![mask_refinement](https://www.dropbox.com/s/v0cgz9xqrwcuzjh/mask_refinement.jpg?raw=1)

- **Mask comparison:** The following figure shows comparison between (i) generated masks and (ii) face segmentations using [YuvalNirkin's FCN netwrok](https://github.com/YuvalNirkin/face_segmentation). Surprisingly, FCN sometimes fails to segment out face occlusions (see the 2nd and 4th rows).

  ![mask_seg_comp](https://www.dropbox.com/s/0tp0fjygfxlofv7/seg_comp.jpg?raw=1)

- **Face detection/tracking using MTCNN and Kalman filter**: More stable detection and smooth tracking.

  ![dlib_vs_MTCNN](https://www.dropbox.com/s/diztxntkss4dt7v/mask_dlib_mtcnn.gif?raw=1)
  
- **V2.1 update:** An improved architecture is updated in order to stablize training. The architecture is greatly inspired by [XGAN](https://arxiv.org/abs/1711.05139) and [MS-D neural network](http://www.pnas.org/content/115/2/254)
  - In v2.1 architecture, we add more discriminators/losses to the GAN. To be specific, they are:
    1. GAN loss for non-masked outputs (common): Add two more discriminators to non-masked outputs.
    2. [Perceptual adversarial loss](https://arxiv.org/abs/1706.09138) (common).
    3. Domain-adversarial loss (XGAN): "It encourages the embeddings learned by the encoder to lie in the same subspace"
    4. Semantic consistency loss (XGAN): Preserve semantic of input.
    5. KL loss (VAE-GAN): KL divergence between  N(0,1) and latent vector.
  - ~~One `res_block` in the decoder is replaced by MS-D network (default depth = 16) for output refinement~~.
    - This is a very inefficient implementation of MS-D network.
  - Preview images are saved in `./previews` folder.
  - (WIP) Random motion blur as data augmentation, reducing ghost effect in output video.
  - FCN8s for face segmentation is introduced to improve masking in video conversion (default `use_FCN_mask = True`).
    - To enable this feature, keras weights file should be generated through jupyter notebook provided in [this repo](https://github.com/shaoanlu/face_segmentation_keras).

## Frequently asked questions

#### 1. Slow video processing / OOM error?
  - It is likely due to too high resolution of input video, modify the parameters in step 13 or 14 will solve it.   
    - First, **increase `video_scaling_offset = 0` to 1** or higher. 
    - If it doesn't help, **set `manually_downscale = True`**.  
    - If the above still do not help, **disable CNN model for face detectoin**.
      ```python
      def process_video(...):
        ...
        #faces = get_faces_bbox(image, model="cnn") # Use CNN model
        faces = get_faces_bbox(image, model='hog') # Use default Haar features.  
      ```
#### 2. How does it work?
  - [This illustration](https://www.dropbox.com/s/4u8q4f03px4spf8/faceswap_GAN_arch3.jpg?raw=1) shows a very high-level and abstract (but not exactly the same) flowchart of the denoising autoencoder algorithm. The objective functions look like [this](https://www.dropbox.com/s/e5j5rl7o3tmw6q0/faceswap_GAN_arch4.jpg?raw=1).
#### 3. No audio in output clips?
  - Set `audio=True` in the video making cell.
    ```python
    output = 'OUTPUT_VIDEO.mp4'
    clip1 = VideoFileClip("INPUT_VIDEO.mp4")
    clip = clip1.fl_image(process_video)
    %time clip.write_videofile(output, audio=True) # Set audio=True
    ```
#### 4. Previews look good, but video result does not seem to transform the face?
  - Default setting transfroms face B to face A.
  - To transform face A to face B, modify the following parameters depending on your current running notebook:
    - Change `path_abgr_A` to `path_abgr_B` in `process_video()` (step 13/14 of v2_train.ipynb and v2_sz128_train.ipynb).
    - Change `whom2whom = "BtoA"` to `whom2whom = "AtoB"` (step 12 of v2_test_video.ipynb).

## Requirements

* keras 2
* Tensorflow 1.3 
* Python 3
* OpenCV
* [moviepy](http://zulko.github.io/moviepy/)
* dlib (optional)
* [face_recognition](https://github.com/ageitgey/face_recognition) (optinoal)

## Acknowledgments
Code borrows from [tjwei](https://github.com/tjwei/GANotebooks), [eriklindernoren](https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/adversarial_autoencoder.py), [fchollet](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb), [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) and [deepfakes](https://pastebin.com/hYaLNg1T). The generative network is adopted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Weights and scripts of MTCNN are from [FaceNet](https://github.com/davidsandberg/facenet). Illustrations are from [irasutoya](http://www.irasutoya.com/).
