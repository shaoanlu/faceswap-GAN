# Notes:
## In this page are notes for my ongoing experiments and failed attmepts.
### 1. BatchNorm/InstanceNorm: 
Caused input/output skin color inconsistency when the 2 training dataset had different skin color dsitribution (light condition, shadow, etc.). But I wonder if this will be solved after further training the model.

### 2. Perceptual loss
Increasing perceptual loss weighting factor (to 1) unstablized training. But the weihgting [.01, .1, .1] I used is not optimal either.

### 3. Bottleneck layers
~~In the encoder architecture, flattening Conv2D and shrinking it to Dense(1024) is crutial for model to learn semantic features, or face representation. If we used Conv layers only (which means larger dimension), will it learn features like visaul descriptors? ([source paper](https://arxiv.org/abs/1706.02932v2), last paragraph of sec 3.1)~~ Similar results can be achieved by replacing the Dense layer with Conv2D strides 2 layers (shrinking feature map to 1x1).

### 4. Transforming Emi Takei to Hinko Sano
Transform Emi Takei to Hinko Sano gave suboptimal results, due to imbalanced training data that over 65% of images of Hinako Sano came from the same video series.

### 5. About mixup and LSGAN
**Mixup** technique ([arXiv](https://arxiv.org/abs/1710.09412)) and **least squares loss** function are adopted ([arXiv](https://arxiv.org/abs/1712.06391)) for training GAN. However, I did not do any ablation experiment on them. Don't know how much impact they had on the outputs.

### 6. Adding landmarks as input feature
Adding face landmarks as the fourth input channel during training (w/ dropout_chance=0.3) force the model to learn(overfit) these face features. However it didn't give me decernible improvement. The following gif is the result clip, it should be mentoined that the landmarks information was not provided during video making, but the model was still able to prodcue accurate landmarks because similar [face, landmarks] pairs are already shown to the model during training.
  - ![landamrks_gif](https://www.dropbox.com/s/ek8y5fued7irq1j/sh_test_clipped4_lms_comb.gif?raw=1)

### 7. **Recursive loop:** Feed model's output image as its input, **repeat N times**.
  - Idea: Since our model is able to transform source face into target face, if we feed generated fake target face as its input, will the model refine the fake face to be more like a real target face?
  - **Version 1 result (w/o alpha mask)** (left to right: source, N=0, N=2, N=10, N=50)
    - ![v1_recur](https://www.dropbox.com/s/hha2w2n4dh49a1k/v1_comb.gif?raw=1)
    - The model seems to refine the fake face (to be more similar with target face), but its shape and color go awry. Furthermore, in certain frames of N=50, **there are blue colors that only appear in target face training data but not source face.** Does this mean that the model is trying to pull out trainnig images it had memoried, or does the mdoel trying to transform the input image into a particular trainnig data?
  - **Version 2 result (w/ alpha mask)** (left to right: source, N=0, N=50, N=150, N=500)
    - ![v2_recur](https://www.dropbox.com/s/zfl8zjlfv2srysx/v2_comb.gif?raw=1)
    - V2 model is more robust. Almost generates the same result before/after applying recursive loop except some artifacts on the bangs.

### 8. **Code manipulation and interpolation**: 
  - ![knn_codes](https://www.dropbox.com/s/a3o1cvqts83h4fl/knn_code_fit.jpg?raw=1)
  - Idea: Refine output face by adding infromation from training images that look like the input image.
  - KNN takes features extracted from ResNet50 model as its input.
  - Similar results can be achieved by simply weighted averaging input image with images retrieved by kNNs (instead of the code).
  - TODO: Implement **alphaGAN**, which integrates VAE that has a more representative latent space.

### 9. **CycleGAN experiment**:
  - ![cyckeGAN exp result](https://www.dropbox.com/s/rj7gi5yft6yw7ng/cycleGAN_exp.JPG?raw=1)
  - Top row: input images.; Bottom row: output images.
  - CycleGAN produces artifacts on output faces. Also, featuers are not consitent before/after transformation, e.g., bangs and skin tone.
  - ~~**CycleGAN with masking**: To be updated.~~

### 10. **(Towards) One Model to Swap Them All**
  - Objective: Train a model that is capable of swapping any given face to Emma Watson.
  - `faceA` folder contains ~2k images of Emma Watson.
  - `faceB` folder contains ~200k images from celebA dataset.
  - Hacks: Add **domain adversaria loss** on embedidngs (from [XGAN](https://arxiv.org/abs/1711.05139) and [this ICCV GAN tutorial](https://youtu.be/uUUvieVxCMs?t=18m59s)). It encourages encoder to generate embbeding from two diffeernt domains to lie in the same subspace (assuming celebA dataset covers almost the true face image dsitribution). Also, heavy data augmentation (random channel shifting, random downsampling, etc.) is applied on face A to pervent overfitting.
  - Result: Model performed poorly on hard sample, e.g., man with beard.

### 11. **Face parts swapping as data augmentation**
  - ![](https://www.dropbox.com/s/1l9n1ple6ymxy8b/data_augm_flowchart.jpg?raw=1)
  - Swap only part of source face (mouth/nose/eyes) to target face, treating the swapped face as a augmented training data for source face.
  - For each source face image, a look-alike target face is retrieved by using knn (taking a averaegd feature map as input) for face part swapping.
  - Result: Unfortunately, the model also learns to generates artifacts as appear in augmented data, e.g., sharp edges around eyes/nose and weirdly warped face. The artifacts of augmented data are caused by non-perfect blending (due to false landmarks and bad perspective warping).

### 12. Neural style transfer as output refinement
  - Problem: The output resolution 64x64 is blurry and sometimes the skin tone does not match the target face. 
  - Question: Is there any other way to refine the 64x64 output face so that it looks natural in, say, a 256x256 input image except increasing output resolution (which leads to much longer training time) or training a super resolution model?
  - Attempts: **Applied neural style transfer techniques as output refinement**. Hoping it can improve output quality and solve color mismatch without additional training of superRes model or increasing model resolution. 
  - Method: We used implementation of neural style transfer from [titu1994/Neural-Style-Transfer](https://github.com/titu1994/Neural-Style-Transfer), [eridgd/WCT-TF](https://github.com/eridgd/WCT-TF), and [jonrei/tf-AdaIN](https://github.com/jonrei/tf-AdaIN). All repos provide pre-trained models. We fed swapped face (i.e., the output image of GAN model) as content image and input face as style image.
  - Results: Style transfer of Gatys et al. gave decent results but require long execution time (~1.5 min per 256x256 image on K80), thus not appplicable for video conversion. The "Universal Style Transfer via Feature Transforms" (WCT) and "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" (AdaIN) somehow failed to preserve the content information (perhaps I did not tune the params well).
  - Conclusion: **Using neural style transfer to improve output quality seems promising**, but we are not sure if it will benefit video quality w/o introducing jitter. Also the execution time is a problem, we should experiment with more arbitrary style transfer networks to see if there is any model that can do a good job on face refinement within one (or several) forward pass(es).
  - ![style_transfer_exp](https://www.dropbox.com/s/r00q5zxojxjofde/style_transfer_comp.png?raw=1)
  
### 13. Model evaluation on Trump/Cage dataset
  - Problem: GANs are hard to evaluate. Generally, Inception Score (IS) and Fréchet Inception Distance (FID score) are the most seen metrics for evaluating the output "reality" (i.e., how close the outputs are to real samples). However, in face-swapping task, we care more about the "quality" of the outputs such as how similar is the transformed output face to its target face. Thus we want to find an objective approach to evauate the model performance as a counter-part of subjectively judging by output visualization.
  - **Evaluation method 1: Compare the predicted identities of VGGFace-ResNet50.** 
    - We look at the predictions of ResNet50 and check if it spits out similar predictions on real/fake images.
    - There are 8631 identities in VGGFace (but unfortunately both Donald Trump and Nicolas Cage are not in this dataset)
    - Top 3 most look-alike identities of "real Trump" are: Alan_Mulally, Jon_Voight, and Tom_Berenger
    - Top 3 most look-alike identities of "fake Trump" are: Alan_Mulally, Franjo_Pooth, and Jon_Voight
    - <img src="https://www.dropbox.com/s/5yg93x9278dguoe/top_1_count_trump.png?raw=1">
    - Top 3 most look-alike identities of "real Cage" are: Jimmy_Stewart, Nick_Grimshaw, and Sylvester_Stallone
    - Top 3 most look-alike identities of "fake Cage" are: Franjo_Pooth, Jimmy_Stewart, and Bob_Beckel
    - <img src="https://www.dropbox.com/s/jz5ovwqqg6rha2s/top_1_count_cage.png?raw=1">
    - **Observation:** Overall, the top-1 look-alike identity of the real Trump/Cage also appear in the top-3 that of the fake one. (Notice that the face-swapping only changes the facial attributes, not the chins and face shape. Thus the fake faces will not look exactly the same with its target face.)
  - **Evaluation method 2: Compare the cosine similarity of extracted VGGFace-ResNet50 features.**
    - Features (embeddings) are extracted from the global average pooling layer (the last layer the before fully-connected layer) of ResNet50, which have diimension of 2048.
    - <img src="https://www.dropbox.com/s/mmzku861gom3j6g/features_umap.png?raw=1" width="450">
    - <img src="https://www.dropbox.com/s/fvij5ckpjyo4iqq/face_feats_vis3d.gif?raw=1">
    - The definition of cosine distance can be found [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html). The cosine similarity is just cosine distance w/o the one minus part.
    - The following 2 heatmaps depict the within-class cosine similarity of real Trump images and real Cage images.
    - <img src="https://www.dropbox.com/s/bb88pjycp6ey7l2/cos_sim_real_trump.png?raw=1" width="350"> <img src="https://www.dropbox.com/s/rgfa7b2zz78x86n/cos_sim_real_cage.png?raw=1" width="350">
    - The following 2 heatmaps illustrate the cosine similarity between real/fake Trump images and between real/fake Cage images. It is obvious that the similarity is not as high as real samples but is still close enough (Note that the low similarity between real and fake Cage is caused by profile faces and heavily occluded faces in real Trump samples, which are hard for the faceswap model to transform.)
    - <img src="https://www.dropbox.com/s/w8zr5ou1s3hw7da/cos_sim_real_trump_fake_trump.png?raw=1" width="350"> <img src="https://www.dropbox.com/s/fy8t1wo2z5eh8bw/cos_sim_real_cage_fake_cage.png?raw=1" width="350">
    - We also checked the cosine similarity between real Trump and real Cage. And the result was not suprising: it shows low similarity between the two identites. This also supports the above observations that the swapped face is much look-alike its target face.
    - <img src="https://www.dropbox.com/s/peydir8ci6rpto4/cos_sim_real_trump_real_cage.png?raw=1" width="350">
    - **Observation:** Evaluation using ResNet50 features demonstrates clear indication that the swapped faces are very look-alike its target face.
  - **Conclusion:** Cosine similarity seems to be a good way to compare performance among different models on the same dataset. Hope this can accelerate our iterations for seaching optimal hyper-parameters and exploring model architectures.
  
### 14. 3D face reconstruction for output refinement
  - Using [PRNet](https://github.com/YadiraF/PRNet) and its accompanying [face-swapping script](https://github.com/YadiraF/PRNet/blob/master/demo_texture.py) to refine the output image.
    - **Result:** For extreme facial expressions, the mouth shape becomes more consistent after face texture editing. (The missing details can be restored through style transfer as shown in exp. 12 above.)
    - Left to right: Input, output, refined output
    - ![3dface01](https://www.dropbox.com/s/dwsj57za9tj127y/3dmodel_refine01.jpg?raw=1)
    - ![3dface02](https://www.dropbox.com/s/fn3sli0gtlb4y78/3dmodel_refine02.jpg?raw=1)
    - For occluded faces, their pose might not be correctly estimated, thus the refined outputs are likely to be distorted. e.g., the displaced microphone in the figure below.
    - ![3dface03](https://www.dropbox.com/s/oaui3vaavv7c9zw/3dmodel_refine03.jpg?raw=1)
