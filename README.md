# faceswap-GAN
Adding Adversarial loss and perceptual loss (VGGface) to deepfakes(reddit user)' auto-encoder architecture.

## Updates
| Date          | Update        |
| ------------- | ------------- |  
| 2018-06-19      | **Readme**: Add previews of the incoming v2.2 model in which introduced an eyes-aware training loss to improve output eyeballs direction.|
| 2018-06-06      | **Model architecture**: Add a self-attention mechanism proposed in [SAGAN](https://arxiv.org/abs/1805.08318) into V2 GAN model. (Note: There is still no official code release for SAGAN, the implementation in this repo. could be wrong. We'll keep an eye on it.)|
| 2018-03-17      | **Training**: V2 model now provides a 40000-iter training schedule which automatically switches to proper loss functions at predefined iterations. ([Cage/Trump dataset results](https://www.dropbox.com/s/24k16vtqkhlf13i/auto_results.jpg?raw=1))| 
| 2018-03-13      | **Model architecture**: V2.1 model now provides 3 base architectures: (i) XGAN, (ii) VAE-GAN, and (iii) a variant of v2 GAN. See "4. Training Phase Configuration" in [v2.1 notebook](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.1_train.ipynb) for detail.| 
| 2018-03-03      | **Model architecture**: Add a [new notebook](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.1_train.ipynb) which contains an improved GAN architecture. The architecture is greatly inspired by [XGAN](https://arxiv.org/abs/1711.05139) and [MS-D neural network](http://www.pnas.org/content/115/2/254).| 
| 2018-02-13      | **Video conversion**: Add a new video procesisng script using **[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)** for face detection. Faster detection with configurable threshold value. No need of CUDA supported dlib. (New notebook: [v2_test_vodeo_MTCNN](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video_MTCNN.ipynb))| 

## Descriptions  
### GAN-v2
* [FaceSwap_GAN_v2_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_train.ipynb) **(recommend for trainnig)**
  - Notebook for training the version 2 GAN model.
  - Video conversion functions are also included.
  
* [FaceSwap_GAN_v2_test_video_MTCNN.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video_MTCNN.ipynb) **(recommend for video conversion)**
  - Notebook for generating videos. Use MTCNN for face detection.
  
* [FaceSwap_GAN_v2_test_video.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video.ipynb)
  - Notebook for generating videos. Use face_recognition module for face detection (requiring dlib package).
  
* [faceswap_WGAN-GP_keras_github.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/temp/faceswap_WGAN-GP_keras_github.ipynb)
  - This notebook is an independent training script for a GAN model of [WGAN-GP](https://arxiv.org/abs/1704.00028). 
  - Perceptual loss is discarded for simplicity. 
  - Not compatible with `_test_video` and `_test_video_MTCNN` notebooks above. 
  - The WGAN-GP model gave similar result with LSGAN model after tantamount (~18k) generator updates.
  - Training can be start easily as the following:
  ```python
  gan = FaceSwapGAN() # instantiate the class
  gan.train(max_iters=10e4, save_interval=500) # start training
  ```
* [FaceSwap_GAN_v2_sz128_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_sz128_train.ipynb)
  - This notebook is an independent script for a model with larger input/output resolution.
  - Not compatible with `_test_video` and `_test_video_MTCNN` notebooks above. 
  - Input and output images have larger shape `(128, 128, 3)`.
  - Introduce minor updates to the architectures: 
    1. Add instance normalization to the generators and discriminators.
    2. Add additional regressoin loss (mae loss) on 64x64 branch output.
  
### Miscellaneous
* [dlib_video_face_detection.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/dlib_video_face_detection.ipynb)
  - Detect/Crop faces in a video using dlib's cnn model. 
  - Pack cropped face images into a zip file.
  
### Training data format 
  - Face images are supposed to be in `./faceA/` or `./faceB/` folder for each taeget respectively. 
  - Face images can be of any size. 
  - For better generalization, source faces can also contain multiple person.

## Generative Adversarial Network for face swapping (version 2)
### 1. Architecture
  ![enc_arch3d](https://www.dropbox.com/s/b43x8bv5xxbo5q0/enc_arch3d_resized2.jpg?raw=1)
  
  ![dec_arch3d](https://www.dropbox.com/s/p09ioztjcxs66ey/dec_3arch3d_resized.jpg?raw=1)
  
  ![dis_arch3d](https://www.dropbox.com/s/szcq8j5axo11mu9/dis_arch3d_resized2.jpg?raw=1)

### 2. Results
- **Improved output quality:** Adversarial loss improves reconstruction quality of generated images.
  ![trump_cage](https://www.dropbox.com/s/24k16vtqkhlf13i/auto_results.jpg?raw=1)

- **Additional results:** [This image](https://www.dropbox.com/s/2nc5guogqk7nwdd/rand_160_2.jpg?raw=1) shows 160 random results generated by v2 GAN with self-attention mechanism (image format: source -> mask -> transformed).

- **Consistent eyeballs direction (v2.2 model):** Results of an updated v2 model which specializes on eyeballs' direcitons are presented below. (Gifs are created using online demo of [DeepWarp](http://163.172.78.19/).) 
  - Top row: v2 model; Bottom row: v2.2 model
  - ![v2_eb](https://www.dropbox.com/s/d0m626ldcw2lop3/v2_comb.gif?raw=1)
  - ![v2.2_eb](https://www.dropbox.com/s/v7wx6r72yfowh98/v2.2_comb.gif?raw=1)

###### The Trump/Cage images are obtained from the reddit user [deepfakes' project](https://pastebin.com/hYaLNg1T) on pastebin.com.

### 3. Features
- **[VGGFace](https://github.com/rcmalli/keras-vggface) perceptual loss:** Perceptual loss improves direction of eyeballs to be more realistic and consistent with input face. It also smoothes out artifacts in the segmentation mask, resulting higher output quality.

- **Attention mask:** Model predicts an attention mask that helps on handling occlusion, eliminating artifacts around edges, and producing natrual skin tone. In below are results transforming Hinako Sano ([佐野ひなこ](https://ja.wikipedia.org/wiki/%E4%BD%90%E9%87%8E%E3%81%B2%E3%81%AA%E3%81%93)) to Emi Takei ([武井咲](https://ja.wikipedia.org/wiki/%E6%AD%A6%E4%BA%95%E5%92%B2)).

  ![mask1](https://www.dropbox.com/s/do3gax2lmhck941/mask_comp1.gif?raw=1)  ![mask2](https://www.dropbox.com/s/gh0yq26qkr31yve/mask_comp2.gif?raw=1)
    - From left to right: source face, swapped face (before masking), swapped face (after masking).

  ![mask_vis](https://www.dropbox.com/s/q6dfllwh71vavcv/mask_vis_rev.gif?raw=1)
    - From left to right: source face, swapped face (after masking), mask heatmap.  
###### Source video: [佐野ひなことすごくどうでもいい話？(遊戯王)](https://www.youtube.com/watch?v=tzlD1CQvkwU)
  
- **Optional 128x128 input/output resolution**: Increase input and output resilution from 64x64 to 128x128.

- **Face detection/tracking using MTCNN and Kalman filter during video conversion**: 
  - MTCNN provides more stable detections. 
  - Kalman filter is introduced to smoothen face bounding box positions over frames and eliminate jitter on the swapped face. 

  ![dlib_vs_MTCNN](https://www.dropbox.com/s/diztxntkss4dt7v/mask_dlib_mtcnn.gif?raw=1)

- **Training schedule**: V2 model provides a predefined training schedule. The Trump/Cage results above are generated by model trained for 21k iters using `TOTAL_ITERS = 30000` predefined training schedule.
  - Training trick: Swapping the decoders in the late stage of training reduces artifacts caused by the extreme facial expressions. E.g., some of the failure cases (of results above) having their mouth open wide are better transformed using this trick.
  
  ![self_attn_and_dec_swapping](https://www.dropbox.com/s/ekpa3caq921v6vk/SA_and_dec_swap2.jpg?raw=1)

### 4. Experimental models
- **V2.1 model:** An improved architecture is updated in order to stablize training. The architecture is greatly inspired by [XGAN](https://arxiv.org/abs/1711.05139) ~~and [MS-D neural network](http://www.pnas.org/content/115/2/254)~~. (Note: V2.1 script is experimental and not well-maintained)
  - V2.1 model  provides three base architectures: (i) XGAN, (ii) VAE-GAN, and (iii) a variant of v2 GAN. (default `base_model="GAN"`)
  - Add more discriminators/losses to the GAN. To be specific, they are:
    1. GAN loss for non-masked outputs (common): Add two more discriminators to non-masked outputs.
    2. [Perceptual adversarial loss](https://arxiv.org/abs/1706.09138) (common): Feature level L1 loss which improves semantic detail.
    3. Domain-adversarial loss (XGAN): "It encourages the embeddings learned by the encoder to lie in the same subspace"
    4. Semantic consistency loss (XGAN): Loss of cosine distance of embeddings to preserve semantic of input.
    5. KL loss (VAE-GAN): KL divergence between  N(0,1) and latent vector.
  - ~~One `res_block` in the decoder is replaced by MS-D network (default depth = 16) for output refinement~~.
    - ~~This is a very inefficient implementation of MS-D network.~~ MS-D network is not included for now.
  - Preview images are saved in `./previews` folder.
  - (WIP) Random motion blur as data augmentation, reducing ghost effect in output video.
  - FCN8s for face segmentation is introduced to improve masking in video conversion (default `use_FCN_mask = True`).
    - To enable this feature, keras weights file should be generated through jupyter notebook provided in [this repo](https://github.com/shaoanlu/face_segmentation_keras).

## Frequently asked questions and troubleshooting

#### 1. How does it work?
  - The following illustration shows a very high-level and abstract (but not exactly the same) flowchart of the denoising autoencoder algorithm. The objective functions look like [this](https://www.dropbox.com/s/e5j5rl7o3tmw6q0/faceswap_GAN_arch4.jpg?raw=1).
  ![flow_chart](https://www.dropbox.com/s/4u8q4f03px4spf8/faceswap_GAN_arch3.jpg?raw=1) 
#### 2. No audio in output clips?
  - Set `audio=True` in the video making cell.
    ```python
    output = 'OUTPUT_VIDEO.mp4'
    clip1 = VideoFileClip("INPUT_VIDEO.mp4")
    clip = clip1.fl_image(process_video)
    %time clip.write_videofile(output, audio=True) # Set audio=True
    ```
#### 3. Previews look good, but it does not transform to the output videos?
  - Default setting transfroms face B to face A.
  - To transform face A to face B, modify the following parameters depending on your current running notebook:
    - Change `path_abgr_A` to `path_abgr_B` in `process_video()` (step 13/14 of v2_train.ipynb and v2_sz128_train.ipynb).
    - Change `direction = "BtoA"` to `direction = "AtoB"` (step 12 of v2_test_video.ipynb).
  - Model performs its full potential when the input images contain less backgrund.
    - Input images should be crop to 80% center area during video conversion. (I'm not sure if this preprocessing step is introduced to GAN model in deepfakes/faceswap)
    - ![readme_note001](https://www.dropbox.com/s/a1kjy0ynnlj2g4c/readme_note00.jpg?raw=1)

## Requirements

* keras 2
* Tensorflow 1.3 
* Python 3
* OpenCV
* [keras-vggface](https://github.com/rcmalli/keras-vggface)
* [moviepy](http://zulko.github.io/moviepy/)
* dlib (optional)
* [face_recognition](https://github.com/ageitgey/face_recognition) (optinoal)

## Acknowledgments
Code borrows from [tjwei](https://github.com/tjwei/GANotebooks), [eriklindernoren](https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/adversarial_autoencoder.py), [fchollet](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb), [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) and [reddit user deepfakes' project](https://pastebin.com/hYaLNg1T). The generative network is adopted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Weights and scripts of MTCNN are from [FaceNet](https://github.com/davidsandberg/facenet). Illustrations are from [irasutoya](http://www.irasutoya.com/).
