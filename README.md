# faceswap-GAN
Adding Adversarial loss and perceptual loss (VGGface) to deepfakes'(reddit user) auto-encoder architecture.

## Updates
| Date          | Update        |
| ------------- | ------------- |  
| 2018-07-25      | **Data preparation:** Add a [new notebook](https://github.com/shaoanlu/faceswap-GAN/blob/master/MTCNN_video_face_detection_alignment.ipynb) for video pre-processing in which MTCNN is used for face detection as well as face alignment. **Evaluation**: Analysis regarding the model output quality on Trump/Cage dataset has been added [here](https://github.com/shaoanlu/faceswap-GAN/blob/master/notes/README.md#13-model-evaluation-for-trumpcage-dataset).|      
| 2018-07-04      | **GAN training**: Add the [relativistic discriminator](https://arxiv.org/abs/1807.00734) as an alternative option to the default mixup training method. Set `loss_config["gan_training"]="relativistic_avg_LSGAN"` in config cells to enable it.|
| 2018-06-29      | **Model architecture**: faceswap-GAN v2.2 now supports different output resolutions: 64x64, 128x128, and 256x256. Default `RESOLUTION = 64` can be changed in the config cell of [v2.2 notebook](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_train_test.ipynb).|
| 2018-06-25      | **New version**: faceswap-GAN v2.2 has been released. The main improvements of v2.2 model are its capability of generating realistic and consistent eye movements (results are shown below, or Ctrl+F for eyes), as well as higher video quality with face alignment.|
| 2018-06-06      | **Model architecture**: Add a self-attention mechanism proposed in [SAGAN](https://arxiv.org/abs/1805.08318) into V2 GAN model. (Note: There is still no official code release for SAGAN, the implementation in this repo. could be wrong. We'll keep an eye on it.)|

## Descriptions  
### faceswap-GAN v2.2 (Recommended model)
* [FaceSwap_GAN_v2.2_train_test.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_train_test.ipynb)
  - Notebook for model training and video conversion of GAN model version 2.2.
  - Additional training images generated through [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb) are required.
  - Supported output resolutions: 64x64, 128x128, and 256x256.
  - Face alignment using 5-points landmarks is introduced to video conversion.
  
* [FaceSwap_GAN_v2.2_video_conversion.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_video_conversion.ipynb)
  - Notebook for video conversion of GAN model version 2.2.
  - Face alignment using 5-points landmarks is introduced to video conversion.
  
* [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb)
  - Notebook for training data preprocessing. It generates binary masks for each training image.
  - Require [face_alignment](https://github.com/1adrianb/face-alignment) package.

### faceswap-GAN v2
* [FaceSwap_GAN_v2_train.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_train.ipynb)
  - Notebook for training the version 2 GAN model.
  - Video conversion functions are also included.
  
* [FaceSwap_GAN_v2_test_video_MTCNN.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_test_video_MTCNN.ipynb)
  - Notebook for generating videos. Use MTCNN for face detection.

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
* [MTCNN_video_face_detection_alignment.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/MTCNN_video_face_detection_alignment.ipynb)
  - This notebook performs face detection on the input video. 
  - Images are save in folder `./faces/raw_faces` and `./faces/aligned_faces` for non-aligned/aligned faces respectively.
  
### Training data format 
  - Face images are supposed to be in `./faceA/` or `./faceB/` folder for each taeget respectively. 
  - Face images can be of any size. 
  - For better generalization, source faces can also be from multiple people.

## Generative adversarial networks for face swapping
### 1. Architecture
  ![enc_arch3d](https://www.dropbox.com/s/b43x8bv5xxbo5q0/enc_arch3d_resized2.jpg?raw=1)
  
  ![dec_arch3d](https://www.dropbox.com/s/p09ioztjcxs66ey/dec_3arch3d_resized.jpg?raw=1)
  
  ![dis_arch3d](https://www.dropbox.com/s/szcq8j5axo11mu9/dis_arch3d_resized2.jpg?raw=1)

### 2. Results
- **Improved output quality:** Adversarial loss improves reconstruction quality of generated images.
  ![trump_cage](https://www.dropbox.com/s/24k16vtqkhlf13i/auto_results.jpg?raw=1)

- **Additional results:** [This image](https://www.dropbox.com/s/2nc5guogqk7nwdd/rand_160_2.jpg?raw=1) shows 160 random results generated by v2 GAN with self-attention mechanism (image format: source -> mask -> transformed).

- **Consistent eye movements (v2.2 model):** Results of the v2.2 model which specializes on eye direcitons are presented below. V2.2 model generated more realistic eyes within shorter iteations. (Input gifs are created using [DeepWarp](http://163.172.78.19/).) 
  - Top row: v2 model; Bottom row: v2.2 model. In column 1, 3, and 5 show input gifs.
  - ![v2_eb](https://www.dropbox.com/s/d0m626ldcw2lop3/v2_comb.gif?raw=1)
  - ![v2.2_eb](https://www.dropbox.com/s/v7wx6r72yfowh98/v2.2_comb.gif?raw=1)

- **Evaluations:** Evaluations for the output quality on Trump/Cage dataset can be found [here](https://github.com/shaoanlu/faceswap-GAN/blob/master/notes/README.md#13-model-evaluation-for-trumpcage-dataset).

###### The Trump/Cage images are obtained from the reddit user [deepfakes' project](https://pastebin.com/hYaLNg1T) on pastebin.com.

### 3. Features
- **[VGGFace](https://github.com/rcmalli/keras-vggface) perceptual loss:** Perceptual loss improves direction of eyeballs to be more realistic and consistent with input face. It also smoothes out artifacts in the segmentation mask, resulting higher output quality.

- **Attention mask:** Model predicts an attention mask that helps on handling occlusion, eliminating artifacts around edges, and producing natrual skin tone. In below are results transforming Hinako Sano ([佐野ひなこ](https://ja.wikipedia.org/wiki/%E4%BD%90%E9%87%8E%E3%81%B2%E3%81%AA%E3%81%93)) to Emi Takei ([武井咲](https://ja.wikipedia.org/wiki/%E6%AD%A6%E4%BA%95%E5%92%B2)).

  ![mask1](https://www.dropbox.com/s/do3gax2lmhck941/mask_comp1.gif?raw=1)  ![mask2](https://www.dropbox.com/s/gh0yq26qkr31yve/mask_comp2.gif?raw=1)
    - From left to right: source face, swapped face (before masking), swapped face (after masking).

  ![mask_vis](https://www.dropbox.com/s/q6dfllwh71vavcv/mask_vis_rev.gif?raw=1)
    - From left to right: source face, swapped face (after masking), mask heatmap.  
###### Source video: [佐野ひなことすごくどうでもいい話？(遊戯王)](https://www.youtube.com/watch?v=tzlD1CQvkwU)

- **Configurable input/output resolution**: The model supports 64x64, 128x128, and 256x256 outupt resolutions.

- **Face tracking/alignment using MTCNN and Kalman filter during video conversion**: 
  - MTCNN provides more stable detections. 
  - Kalman filter is introduced to smoothen face bounding box positions over frames and eliminate jitter on the swapped face.
  - Face alignment (FA) further stabilises the output results.

  ![dlib_vs_MTCNN](https://www.dropbox.com/s/diztxntkss4dt7v/mask_dlib_mtcnn.gif?raw=1)  
  ![comp_FA](https://www.dropbox.com/s/kviue4065gdqfnt/comp_fa.gif?raw=1)

- **Training schedule**: V2 model provides a predefined training schedule. The Trump/Cage results above are generated by model trained for 21k iters using `TOTAL_ITERS = 30000` predefined training schedule.
  - Training trick: Swapping the decoders in the late stage of training reduces artifacts caused by the extreme facial expressions. E.g., some of the failure cases (of results above) having their mouth open wide are better transformed using this trick.
  
  ![self_attn_and_dec_swapping](https://www.dropbox.com/s/ekpa3caq921v6vk/SA_and_dec_swap2.jpg?raw=1)
  
- **Eyes-aware training:** Introduce reconstruction loss and edge loss around eyes area, which guide the model to generate realistic eyes.

### 4. Experimental models
- **V2.1 model:** An improved architecture is updated in order to stablize training. The architecture is greatly inspired by [XGAN](https://arxiv.org/abs/1711.05139) ~~and [MS-D neural network](http://www.pnas.org/content/115/2/254)~~. (Note: V2.1 script is experimental and not well-maintained)
  - V2.1 model  provides three base architectures: (i) XGAN, (ii) VAE-GAN, and (iii) a variant of v2 GAN. (default `base_model="GAN"`)
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
  - Model performs its full potential when the input images contain less backgrund.
    - Input images should better be preprocessed with face alignment methods.
    - ![readme_note001](https://www.dropbox.com/s/a1kjy0ynnlj2g4c/readme_note00.jpg?raw=1)

## Requirements

* keras 2.1.5
* Tensorflow 1.6.0 
* Python 3.6.4
* OpenCV
* [keras-vggface](https://github.com/rcmalli/keras-vggface)
* [moviepy](http://zulko.github.io/moviepy/)
* [prefetch_generator](https://github.com/justheuristic/prefetch_generator) (required for v2.2 model)
* [face-alignment](https://github.com/1adrianb/face-alignment) (required as preprocessing for v2.2 model)
* dlib (optional)
* [face_recognition](https://github.com/ageitgey/face_recognition) (optinoal)

## Acknowledgments
Code borrows from [tjwei](https://github.com/tjwei/GANotebooks), [eriklindernoren](https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/adversarial_autoencoder.py), [fchollet](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb), [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) and [reddit user deepfakes' project](https://pastebin.com/hYaLNg1T). The generative network is adopted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Weights and scripts of MTCNN are from [FaceNet](https://github.com/davidsandberg/facenet). Illustrations are from [irasutoya](http://www.irasutoya.com/).
