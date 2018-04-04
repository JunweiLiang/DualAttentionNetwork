# Dual attention network

This repository contains the code and models for this CVPR 2017 paper (image-to-text and text-to-image task):

	Hyeonseob Nam, Jung-Woo Ha, and Jeonghee Kim. 
	"Dual attention networks for multimodal reasoning and matching." 
	in Proc. CVPR 2017

Thanks to instructions from the author (Hyeonseob Nam), I was able to (almost) reproduce the number reported in the paper on Flickr30k:

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