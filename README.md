# ModelScalingBench
**title**

The repository contains the data extracted from the empirical study (submitted to the Future Internet journal) showcasing the efficacy of our Deep Neural Network model scaling benchmark framework.

This data can be used for reproducing the results of our work, or be used as workload input for other studies.

## Models and Datasets

For the empirical study we embraced the following open and publicly available DNN model structures and datasets as workloads:

- [BERT](https://github.com/google-research/bert) first 20 models (up to L=10) made available by Google Research and the [Glue-MRPC](https://paperswithcode.com/dataset/mrpc) dataset made available by Microsoft Research.
- [EfficientNet](https://keras.io/api/applications/efficientnet/) B0-B5 variants made available by Keras and the [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) dataset hosted on Hugging Face.
- Multi-Layer Perceptrons, 120 in total, trained with the [California Housing dataset](https://keras.io/api/datasets/california_housing/) from StatLib and provided via Keras for regression analysis. The script and notebook generating the MLPs is added and available in the repo.
