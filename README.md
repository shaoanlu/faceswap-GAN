# deepfakes-faceswap-GAN
A GAN model that is built upon deepfakes' autoencoder, adding adversarial loss and perceptual loss (VGGface) to improve output result.

# [Jupyter notebook WIP]
WIP code cleanup.

# Results

Transform Hinako Sano [(佐野ひなこ)](https://ja.wikipedia.org/wiki/%E4%BD%90%E9%87%8E%E3%81%B2%E3%81%AA%E3%81%93) to Emi Takei [(武井咲)](https://ja.wikipedia.org/wiki/%E6%AD%A6%E4%BA%95%E5%92%B2).  

## 1. [deepfakes' autorecoder](https://github.com/deepfakes/faceswap) (non official repo.)

(The result of autoencoder can be much better if we trained longer.)

![AE GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/AE_sh_test.gif)

## 2. GAN (adding adversarial loss)

Adversarial loss improves output resolution.

![GAN_GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/woPL_sh_test3.gif)

## 3. GAN (adding adversarial loss and [VGGface](https://github.com/rcmalli/keras-vggface) perceptual loss)

By applying perceptual loss, the movemnet of eyeballs becomes more realistic (although hard to distinguish in the gifs).

![GAN_PL_GIF](https://github.com/shaoanlu/faceswap-GAN/raw/master/gifs/PL_sh_test3.gif)

###### Source video: [佐野ひなことすごくどうでもいい話？(遊戯王)](https://www.youtube.com/watch?v=tzlD1CQvkwU)

# Requirements

* keras 2
* Tensorflow 1.3 
* Python 3
* dlib
* face_recodnition
* moviepy
