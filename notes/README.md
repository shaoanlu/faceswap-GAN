# Notes:
## In this page are notes for my ongoing experiments and failed attmeps.
1. **BatchNorm/InstanceNorm**: Caused input/output skin color inconsistency when the 2 training dataset had different skin color dsitribution (light condition, shadow, etc.). But I wonder if this will be solved after further training the model.
2. Increasing perceptual loss weighting factor (to 1) unstablized training. But the weihgting [.01, .1, .1] I used is not optimal either.
3. ~~In the encoder architecture, flattening Conv2D and shrinking it to Dense(1024) is crutial for model to learn semantic features, or face representation. If we used Conv layers only (which means larger dimension), will it learn features like visaul descriptors? ([source paper](https://arxiv.org/abs/1706.02932v2), last paragraph of sec 3.1)~~ Similar results can be achieved by replacing the Dense layer with Conv2D strides 2 layers (shrinking feature map to 1x1).
4. Transform Emi Takei to Hinko Sano gave suboptimal results, due to imbalanced training data that over 65% of images of Hinako Sano came from the same video series.
5. **Mixup** technique ([arXiv](https://arxiv.org/abs/1710.09412)) and **least squares loss** function are adopted ([arXiv](https://arxiv.org/abs/1712.06391)) for training GAN. However, I did not do any ablation experiment on them. Don't know how much impact they had on outputs.
6. **Adding face landmarks** as the fourth input channel during training (w/ dropout_chance=0.3) force the model to learn(overfit) these face features. However it didn't give me decernible improvement. The following gif is the result clip, it should be mentoined that the landmarks information was not provided during video making, but the model was still able to prodcue accurate landmarks because similar [face, landmarks] pairs are already shown to the model during training.
  - ![landamrks_gif](https://www.dropbox.com/s/ek8y5fued7irq1j/sh_test_clipped4_lms_comb.gif?raw=1)

7. **Recursive loop:** Feed model's output image back as its input, **repeat N times**.
  - Idea: Since our model is able to transform source face into target face, if we feed generated fake target face as its input, will the model refine the fake face to be more like a real target face?
  - **Version 1 result** (left to right: source, No recursion, N=2, N=10, N=50)
    - ![v1_recur](https://www.dropbox.com/s/hha2w2n4dh49a1k/v1_comb.gif?raw=1)
    - The model seems to refine the fake face (to be more similar with target face), but its shape and color go awry. Furthermore, in certain frames of N=50, **there are blue colors that only appear in target face training data but oot source face.** Does this mean that the model is trying to pull out trainnig images is had memoried, or does the mdoel trying to transform the input image into a certain trainnig data?
  - **Version 2 result** (left to right: source, No recursion, N=50, N=150, N=500)
    - ![v2_recur](https://www.dropbox.com/s/zfl8zjlfv2srysx/v2_comb.gif?raw=1)
    - V2 model is more robust. Almost generates the same result before/after applying recursive loop except some artifacts on the bangs.
8. **Code manipulation**: 
  - ![knn_codes](https://www.dropbox.com/s/a3o1cvqts83h4fl/knn_code_fit.jpg?raw=1)
  - Idea: Refine output face by adding infromation from training images that look like the input image.
  - Similar results can be achieved by simply weighted averaging input image with images retrieved by kNNs (instead of the code).
  - TODO: Implement **alphaGAN**, which integrates VAE that has a more representative latent space.
9. **CycleGAN experiment**:
  - ![cyckeGAN exp result](https://www.dropbox.com/s/rj7gi5yft6yw7ng/cycleGAN_exp.JPG?raw=1)
  - Top row: input images.; Bottom row: output images.
  - CycleGAN produces artifacts on output faces. Also, featuers are not consitent before/after transformation, e.g., bangs and skin tone.
9.5. **CycleGAN with masking**
  - To be updated.
10. **(Towards) One Model to Swap Them All**
  - Objective: Train a model that is capable of swapping any given face to Emma Watson.
  - `faceA` folder contains ~2k images of Emma Watson.
  - `faceB` folder contains ~200k images from celebA dataset.
  - Hacks: Add **domain adversaria loss** on embedidngs (idea from [XGAN](https://arxiv.org/pdf/1711.05139.pdf) and [an ICCV GAN tutorial](https://youtu.be/uUUvieVxCMs?t=18m59s)). It encourages encoder to generate embbeding from two diffeernt domains to lie in the same subspace (assuming celebA dataset covers almost the true face image dsitribution). Also, heavy data augmentation (random channel shifting, random downsampling, etc.) is applied on face A to pervent overfitting.
  - Result: Model performs poorly on hard sample, e.g., man with beard.
