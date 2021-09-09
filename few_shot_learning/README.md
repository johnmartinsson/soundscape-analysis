# Few-shot learning

A collection of few-shot learning code.

In a __K__-way __N__-shot problem we have access to a support set with __N__ examples in some space __X__ of each of the __K__ classes, and we want to learn how to classify unseen query points __q__ from the same space __X__ from these few examples. 

A typical approach for few-shot learning is to learn a feature extractor __r = f(x)__ where __r__ is the representation of the input __x__, and assign the class of the query point __q__ according to closeness to the support of each class in this representation space. E.g., nearest neighbour for the mean representation of each class support.

## Useful libraries and github repos
- [few-shot](https://github.com/oscarknagg/few-shot)
  - implementations of various few-shot learning methods
- [few-shot bioacoustic baseline](https://github.com/c4dm/dcase-few-shot-bioacoustic/tree/main/baselines/deep_learning)
  - baseline implementation of prototypical networks used in [DCASE_2021_task5](http://dcase.community/challenge2021/task-few-shot-bioacoustic-event-detection)

## Ideas
- Capture the variance in the representation space. Early methods in few-shot learning, such as prototypical networks, represent the support of a class as the average of the support in the representation space. It seems reasonable to assume that the variance of the representations for each support class is different, i.e.,g that they are more or less spread out around their average. Papers which seem to have explored this idea are:
  - [Infinite Mixture Prototypes for Few-shot Learning](http://proceedings.mlr.press/v97/allen19b.html)
  - [Variational Few-Shot Learning](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Variational_Few-Shot_Learning_ICCV_2019_paper.html)
  - I did not know that people had done this already, and I like these ideas. Is there anything else to do in this direction? Or is it exhausted.
