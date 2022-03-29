# Doodle It Yourself: Class Incremental Learning by Drawing a Few Sketches, CVPR 2022.
**Ayan Kumar Bhunia**, Viswanatha Reddy Gajjala, Subhadeep Koley, Rohit Kundu, Aneeshan Sain, Tao Xiang , Yi-Zhe Song, “Doodle It Yourself: Class Incremental Learning by Drawing a Few Sketches”, IEEE Conf. on Computer Vision and Pattern Recognition (**CVPR**), 2022.

*Code coming soon ....*


## Abstract
The human visual system is remarkable in learning new visual concepts from just a few examples. This is precisely the goal behind few-shot class incremental learning (FSCIL), where the emphasis is additionally placed on ensuring the model does not suffer from "forgetting". In this paper, we push the boundary further for FSCIL by addressing two key questions that bottleneck its ubiquitous application (i) can the model learn from diverse modalities other than just photo (as humans do), and (ii) what if photos are not readily accessible (due to ethical and privacy constraints). Our key innovation lies in advocating the use of sketches as a new modality for class support. The product is a "Doodle It Yourself" (DIY) FSCIL framework where the users can freely sketch a few examples of a novel class for the model to learn to recognize photos of that class. For that, we present a framework that infuses (i) gradient consensus for domain invariant learning, (ii) knowledge distillation for preserving old class information, and (iii) graph attention networks for message passing between old and novel classes. We experimentally show that sketches are better class support than text in the context of FSCIL, echoing findings elsewhere in the sketching literature.


## Citation

If you find this article useful in your research, please consider citing:
```
@InProceedings{DoodleIncremental,
author = {Ayan Kumar Bhunia and Viswanatha Reddy Gajjala and Subhadeep Koley and Rohit Kundu and Aneeshan Sain and Tao Xiang and Yi-Zhe Song},
title = {Doodle It Yourself: Class Incremental Learning by Drawing a Few Sketches},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2022}
}
```
## Work done at [SketchX Lab](http://sketchx.ai/), CVSSP, University of Surrey. 
