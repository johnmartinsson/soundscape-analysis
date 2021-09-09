# Few-shot learning

A collection of few-shot learning code.

In a __K__-way __N__-shot problem we have access to a support set with __N__ examples in some space __X__ of each of the __K__ classes. A typical approach for few-shot learning is to learn a feature extractor __f : X -> R__, e.g., __r = f(x)__ where __r__ is the representation of the input __x__, and then 

## Useful libraries and github repos
- [few-shot](https://github.com/oscarknagg/few-shot)
  - implementations of various few-shot learning methods
- [few-shot bioacoustic baseline](https://github.com/c4dm/dcase-few-shot-bioacoustic/tree/main/baselines/deep_learning)
  - baseline implementation of prototypical networks used in [DCASE_2021_task5](http://dcase.community/challenge2021/task-few-shot-bioacoustic-event-detection)
