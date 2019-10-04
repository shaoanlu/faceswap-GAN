# faceswap-GAN
Adding Adversarial loss and perceptual loss (VGGface) to deepfakes'(reddit user) auto-encoder architecture.

## Updates
| Date          | Update        |
| ------------- | ------------- |    
| 2018-08-27      | **Colab support:** A [colab notebook](https://colab.research.google.com/github/shaoanlu/faceswap-GAN/blob/master/colab_demo/faceswap-GAN_colab_demo.ipynb) for faceswap-GAN v2.2 is provided.| 
| 2018-07-25      | **Data preparation:** Add a [new notebook](https://github.com/shaoanlu/faceswap-GAN/blob/master/MTCNN_video_face_detection_alignment.ipynb) for video pre-processing in which MTCNN is used for face detection as well as face alignment.| 
| 2018-06-29      | **Model architecture**: faceswap-GAN v2.2 now supports different output resolutions: 64x64, 128x128, and 256x256. Default `RESOLUTION = 64` can be changed in the config cell of [v2.2 notebook](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_train_test.ipynb).|
| 2018-06-25      | **New version**: faceswap-GAN v2.2 has been released. The main improvements of v2.2 model are its capability of generating realistic and consistent eye movements (results are shown below, or Ctrl+F for eyes), as well as higher video quality with face alignment.|
| 2018-06-06      | **Model architecture**: Add a self-attention mechanism proposed in [SAGAN](https://arxiv.org/abs/1805.08318) into V2 GAN model. (Note: There is still no official code release for SAGAN, the implementation in this repo. could be wrong. We'll keep an eye on it.)|

## Google Colab support
Here is a [playground notebook](https://colab.research.google.com/github/shaoanlu/faceswap-GAN/blob/master/colab_demo/faceswap-GAN_colab_demo.ipynb) for faceswap-GAN v2.2 on Google Colab. Users can train their own model in the browser.

[Update 2019/10/04] There seems to be import errors in the latest Colab environment due to inconsistent version of packages. Please make sure that the Keras and TensorFlow follow the version number shown in the requirement section below.

## Descriptions  
### faceswap-GAN v2.2
* [FaceSwap_GAN_v2.2_train_test.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_train_test.ipynb)
  - Notebook for model training of faceswap-GAN model version 2.2.
  - This notebook also provides code for still image transformation at the bottom.
  - Require additional training images generated through [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb).
  
* [FaceSwap_GAN_v2.2_video_conversion.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_video_conversion.ipynb)
  - Notebook for video conversion of faceswap-GAN model version 2.2.
  - Face alignment using 5-points landmarks is introduced to video conversion.
  
* [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb)
  - Notebook for training data preprocessing. Output binary masks are save in `./binary_masks/faceA_eyes` and `./binary_masks/faceB_eyes` folders.
  - Require [face_alignment](https://github.com/1adrianb/face-alignment) package. (An alternative method for generating binary masks (not requiring `face_alignment` and `dlib` packages) can be found in [MTCNN_video_face_detection_alignment.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/MTCNN_video_face_detection_alignment.ipynb).) 
  
* [MTCNN_video_face_detection_alignment.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/MTCNN_video_face_detection_alignment.ipynb)
  - This notebook performs face detection/alignment on the input video. 
  - Detected faces are saved in `./faces/raw_faces` and `./faces/aligned_faces` for non-aligned/aligned results respectively.
  - Crude eyes binary masks are also generated and saved in `./faces/binary_masks_eyes`. These binary masks can serve as a suboptimal alternative to masks generated through [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb). 
  
**Usage**
1. Run [MTCNN_video_face_detection_alignment.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/MTCNN_video_face_detection_alignment.ipynb) to extract faces from videos. Manually move/rename the aligned face images into `./faceA/` or `./faceB/` folders.
2. Run [prep_binary_masks.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/prep_binary_masks.ipynb) to generate binary masks of training images. 
    - You can skip this pre-processing step by (1) setting `use_bm_eyes=False` in the config cell of the train_test notebook, or (2) use low-quality binary masks generated in step 1.
3. Run [FaceSwap_GAN_v2.2_train_test.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_train_test.ipynb) to train  models.
4. Run  [FaceSwap_GAN_v2.2_video_conversion.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2.2_video_conversion.ipynb) to create videos using the trained models in step 3. 
  
### Miscellaneous
* [faceswap-GAN_colab_demo.ipynb](https://github.com/shaoanlu/faceswap-GAN/blob/master/colab_demo/faceswap-GAN_colab_demo.ipynb)
  - An all-in-one notebook for demostration purpose that can be run on Google colab.
  
### Training data format 
  - Face images are supposed to be in `./faceA/` or `./faceB/` folder for each taeget respectively. 
  - Images will be resized to 256x256 during training.

## Generative adversarial networks for face swapping
### 1. Architecture
  ![enc_arch3d](https://www.dropbox.com/s/b43x8bv5xxbo5q0/enc_arch3d_resized2.jpg?raw=1)
  
  ![dec_arch3d](https://www.dropbox.com/s/p09ioztjcxs66ey/dec_3arch3d_resized.jpg?raw=1)
  
  ![dis_arch3d](https://www.dropbox.com/s/szcq8j5axo11mu9/dis_arch3d_resized2.jpg?raw=1)

### 2. Results
- **Improved output quality:** Adversarial loss improves reconstruction quality of generated images.
  ![trump_cage](https://www.dropbox.com/s/24k16vtqkhlf13i/auto_results.jpg?raw=1)

- **Additional results:** [This image](https://www.dropbox.com/s/2nc5guogqk7nwdd/rand_160_2.jpg?raw=1) shows 160 random results generated by v2 GAN with self-attention mechanism (image format: source -> mask -> transformed).

- **Evaluations:** Evaluations of the output quality on Trump/Cage dataset can be found [here](https://github.com/shaoanlu/faceswap-GAN/blob/master/notes/README.md#13-model-evaluation-for-trumpcage-dataset).

###### The Trump/Cage images are obtained from the reddit user [deepfakes' project](https://pastebin.com/hYaLNg1T) on pastebin.com.

### 3. Features
- **[VGGFace](https://github.com/rcmalli/keras-vggface) perceptual loss:** Perceptual loss improves direction of eyeballs to be more realistic and consistent with input face. It also smoothes out artifacts in the segmentation mask, resulting higher output quality.

- **Attention mask:** Model predicts an attention mask that helps on handling occlusion, eliminating artifacts, and producing natrual skin tone.

- **Configurable input/output resolution (v2.2)**: The model supports 64x64, 128x128, and 256x256 outupt resolutions.

- **Face tracking/alignment using MTCNN and Kalman filter in video conversion**: 
  - MTCNN is introduced for more stable detections and reliable face alignment (FA). 
  - Kalman filter smoothen the bounding box positions over frames and eliminate jitter on the swapped face.
  ![comp_FA](https://www.dropbox.com/s/kviue4065gdqfnt/comp_fa.gif?raw=1)
  
- **Eyes-aware training:** Introduce high reconstruction loss and edge loss in eyes area, which guides the model to generate realistic eyes.

## Frequently asked questions and troubleshooting

#### 1. How does it work?
  - The following illustration shows a very high-level and abstract (but not exactly the same) flowchart of the denoising autoencoder algorithm. The objective functions look like [this](https://www.dropbox.com/s/e5j5rl7o3tmw6q0/faceswap_GAN_arch4.jpg?raw=1).
  ![flow_chart](https://www.dropbox.com/s/4u8q4f03px4spf8/faceswap_GAN_arch3.jpg?raw=1) 
#### 2. Previews look good, but it does not transform to the output videos?
  - Model performs its full potential when the input images are preprocessed with face alignment methods.
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

## Acknowledgments
Code borrows from [tjwei](https://github.com/tjwei/GANotebooks), [eriklindernoren](https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/adversarial_autoencoder.py), [fchollet](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.5-introduction-to-gans.ipynb), [keras-contrib](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py) and [reddit user deepfakes' project](https://pastebin.com/hYaLNg1T). The generative network is adopted from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Weights and scripts of MTCNN are from [FaceNet](https://github.com/davidsandberg/facenet). Illustrations are from [irasutoya](http://www.irasutoya.com/).
