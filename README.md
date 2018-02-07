# faceswap-GAN
Adding Adversarial loss and perceptual loss (VGGface) to deepfakes' auto-encoder architecture.

## News
| Date          | Update        |
| ------------- | ------------- | 
| 2018-02-07      | **Video-making**: Auto downscale image resolution for face detection, preventing OOM error. This does not affect output video resolution. (Target notebook: [v2_sz128_train](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynb), [v2_train](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_train.ipynb), and [v2_test_video](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video.ipynb))| 

## Descriptions
### GAN-v1
* [FaceSwap_GAN_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_github.ipynb)

  1. Build and train a GAN model. 
  2. Use moviepy module to output a video clip with swapped face.  
  
### GAN-v2
* [FaceSwap_GAN_v2_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_train.ipynb): Detailed training procedures can be found in this notebook.
  1. Build and train a GAN model. 
  2. Use moviepy module to output a video clip with swapped face.
  
* [FaceSwap_GAN_v2_test_img.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_img.ipynb): Provides `swap_face()` function that require less VRAM.
  1. Load trained model.
  2. Do single image face swapping.
  
* [FaceSwap_GAN_v2_test_video.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video.ipynb)
  1. Load trained model.
  2. Use moviepy module to output a video clip with swapped face. 
  
* [faceswap_WGAN-GP_keras_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/temp/faceswap_WGAN-GP_keras_github.ipynb)
  - This notebook contains a class of GAN mdoel using [WGAN-GP](https://arxiv.org/abs/1704.00028). 
  - Perceptual loss is discarded for simplicity. 
  - The WGAN-GP model gave me similar result with LSGAN model after tantamount (~18k) generator updates.
  ```python
  gan = FaceSwapGAN() # instantiate the class
  gan.train(max_iters=10e4, save_interval=500) # start training
  ```
* [FaceSwap_GAN_v2_sz128_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynb)
  - Input and output images have shape `(128, 128, 3)`.
  - Minor updates on the architectures: 
    1. Add instance normalization to generators and discriminators.
    2. Add additional regressoin loss (mae loss) on 64x64 branch output.
  - Not compatible with `_test_video` and `_test_img` notebooks above.
  
### Others
* [dlib_video_face_detection.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/dlib_video_face_detection.ipynb)
  1. Detect/Crop faces in a video using dlib's cnn model. 
  2. Pack cropped face images into a zip file.
 
* Training data: Face images are supposed to be in `./faceA/` and `./faceB/` folder for each target respectively. Face images can be of any size. (Updated 3, Jan., 2018)

## Results

In below are results that show trained models transforming Hinako Sano ([佐野ひなこ](https://ja.wikipedia.org/wiki/%E4%BD%90%E9%87%8E%E3%81%B2%E3%81%AA%E3%81%93)) to Emi Takei ([武井咲](https://ja.wikipedia.org/wiki/%E6%AD%A6%E4%BA%95%E5%92%B2)).  
###### Source video: [佐野ひなことすごくどうでもいい話？(遊戯王)](https://www.youtube.com/watch?v=tzlD1CQvkwU)
### 1. Autorecoder baseline

Autoencoder based on deepfakes' script. It should be mentoined that the result of autoencoder (AE) can be much better if we trained it for longer.

![AE_results](https://www.dropbox.com/s/n9xjzhlc4llbh96/AE_results.png?raw=1)

### 2. Generative Adversarial Network, GAN (version 1)

**Improved output quality:** Adversarial loss improves reconstruction quality of generated images. In addition, when perceptual loss is apllied, the direction of eyeballs becomes more realistic and consistent with input face.

![GAN_PL_results](https://www.dropbox.com/s/ex7z8upst0toyf0/wPL_results_resized.png?raw=1)

**[VGGFace](https://github.com/rcmalli/keras-vggface) perceptual loss (PL):** The following figure shows nuanced eyeballs direction of output faces trained with/without PL. 

![Comp PL](https://www.dropbox.com/s/dszawjl2hlp9mut/comparison_PL_rev.png?raw=1)

**Smoothed bounding box (Smoothed bbox):** Exponential moving average of bounding box position over frames is introduced to eliminate jittering on the swapped face. See the below gif for comparison.

![bbox](https://www.dropbox.com/s/fla8lcpfpb20rt2/bbox_comp_annotated.gif?raw=1)
  - A. Source face.
  - B. Swapped face, using smoothing mask (smoothes edges of output image when pasting it back to input image).
  - C. Swapped face, using smoothing mask and face alignment.
  - D. Swapped face, using smoothing mask and smoothed bounding box.

### 3. Generative Adversarial Network, GAN (version 2)

**Version 1 features:** Most of features in version 1 are inherited, including perceptual loss and smoothed bbox.

**Segmentation mask prediction:** Model learns a proper mask that helps on handling occlusion, eliminating artifacts on bbox edges, and producing natrual skin tone.

![mask0](https://www.dropbox.com/s/iivdpsba1sa7wg1/comp_mask_rev.png?raw=1)

![mask1](https://www.dropbox.com/s/do3gax2lmhck941/mask_comp1.gif?raw=1)  ![mask2](https://www.dropbox.com/s/gh0yq26qkr31yve/mask_comp2.gif?raw=1)
  - Left: Source face.
  - Middle: Swapped face, before masking.
  - Right: Swapped face, after masking.

**Mask visualization**: The following gif shows output mask & face bounding box.

![mask_vis](https://www.dropbox.com/s/q6dfllwh71vavcv/mask_vis_rev.gif?raw=1)
  - Left: Source face.
  - Middle: Swapped face, after masking.
  - Right: Mask heatmap & face bounding box.
  
**Optional 128x128 input/output resolution**: Increase input and output size to 128x128.

**Mask refinement**: Tips for mask refinement are provided in the jupyter notebooks (VGGFace ResNet50 is required). The following figure shows generated masks before/after refinement. Input faces are from [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

![mask_refinement](https://www.dropbox.com/s/v0cgz9xqrwcuzjh/mask_refinement.jpg?raw=1)

## Frequently asked questions

#### 1. Video making is slow / OOM error?
  - It is likely due to too high resolution of input video, try to   
  **Increase `video_scaling_offset = 0`** to 1 or higher (update 2018-02-07),
  
    or
  **disable CNN model for face detectoin** (update 2018-02-07)
    ```python
    def process_video(...):
      ...
      #faces = get_faces_bbox(image, model="cnn") # Use CNN model
      faces = get_faces_bbox(image, model='hog') # Use default Haar features.  
    ```
    or
   **reduce input size**
    ```python
    def porcess_video(input_img):
      # Reszie to 1/2x width and height.
      input_img = cv2.resize(input_img, (input_img.shape[1]//2, input_img.shape[0]//2))
      image = input_image
      ...
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

## Requirements

* keras 2
* Tensorflow 1.3 
* Python 3
* OpenCV
* dlib
* [face_recognition](https://github.com/ageitgey/face_recognition)
* [moviepy](http://zulko.github.io/moviepy/)

## Acknowledgments
Code borrows from [tjwei](https://github.com/tjwei/GANotebooks), [eriklindernoren](https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/adversarial_autoencoder.py), [fchollet](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb), [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) and [deepfakes](https://pastebin.com/hYaLNg1T). The generative network is adopted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Part of illustrations are from [irasutoya](http://www.irasutoya.com/).
