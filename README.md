# Dual attention network

This repository contains the code and models for this CVPR 2017 paper (image-to-text and text-to-image task):

	Hyeonseob Nam, Jung-Woo Ha, and Jeonghee Kim. 
	"Dual attention networks for multimodal reasoning and matching." 
	in Proc. CVPR 2017

Thanks to instructions from the author (Hyeonseob Nam), I was (almost) able to reproduce the number reported in the paper on Flickr30k:

<table>
  <tr>
    <td></td>
    <td colspan="4">Image-to-Text</td>
    <td colspan="4">Text-to-Image</td>
  </tr>
  <tr>
    <td>Method</td>
    <td>R@1</td>
    <td>R@5</td>
    <td>R@10</td>
    <td>MR</td>
    <td>R@1</td>
    <td>R@5</td>
    <td>R@10</td>
    <td>MR</td>
  </tr>
  <tr>
    <td>DAN Paper</td>
    <td>55.0</td>
    <td>81.8</td>
    <td>89.0</td>
    <td>1</td>
    <td>39.4</td>
    <td>69.2</td>
    <td>79.1</td>
    <td>2</td>
  </tr>
  <tr>
    <td>This Implementation</td>
    <td>50.0</td>
    <td>80.0</td>
    <td>88.3</td>
    <td>1.5</td>
    <td>38.4</td>
    <td>70.2</td>
    <td>80.3</td>
    <td>2</td>
  </tr>
</table>


## Dependencies
+ Python 2.7; TensorFlow >= 1.4.0; tqdm and nltk (for preprocessing)
+ Flickr30k Images and Text
+ Dataset splits from [here](https://aladdin1.inf.cs.cmu.edu/shares/splits.tgz). This split is the same as [m-RNN](http://www.stat.ucla.edu/~junhua.mao/m-RNN.html).
+ Pretrained [Resnet-152 Model](http://models.tensorpack.com/ResNet/ImageNet-ResNet152.npz) from Tensorpack

## Training

1. Get Resnet feature
```
$ python resnet-extractor/extract.py flickr30k_images/ ImageNet-ResNet152.npz resnet-152 --batch_size 20 --resize 448 --depth 152
```

2. Preprocess
```
$ python prepro_flickr30k.py splits/ results_20130124.token prepro --noword2vec --noimgfeat
```

3. Training

I use a slightly different training schedule. Batch size 256, learning rate 0.1 and 0.5 dropout for the first 60 epochs and 0.8 dropout and learning rate 0.05 for the next epochs. Also I use Adadelta as optimizer.

```
$ python main.py prepro models dan --no_wordvec --word_emb_size 512 --num_hops 2 --word_count_thres 1 --sent_size_thres 200 --word_size_thres 20 --hidden_size 512 --keep_prob 0.5 --margin 100 --num_epochs 60 --save_period 1000 --batch_size 256 --clip_gradient_norm 0.1 --init_lr 0.1 --wd 0.0005 --featpath resnet-152/ --feat_dim 14,14,2048 --hn_num 32 --is_train
```

4. Testing with the model
You can download [my model](https://aladdin1.inf.cs.cmu.edu/shares/dan_model_04042018.tgz) and put it in models/00/dan/best/ to directly run it
```
$ python main.py prepro models dan --no_wordvec --word_emb_size 512 --num_hops 2 --word_count_thres 1 --sent_size_thres 200 --word_size_thres 20 --hidden_size 512 --keep_prob 0.5 --margin 100 --num_epochs 60 --save_period 1000 --batch_size 256 --clip_gradient_norm 0.1 --init_lr 0.1 --wd 0.0005 --featpath resnet-152/ --feat_dim 14,14,2048 --hn_num 32 --is_test --load_best
```