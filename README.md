# MET : Masked Encoding Tabular Data

This repository is the official implementation of [MET](https://arxiv.org/abs/2206.08564).

```
Disclaimer : This is not an officially supported Google product.
```

![Architecture](./MET.png)
## Requirements

To run experiments mentioned in the paper and install requirements use python version >=3.7:

```setup
git clone http://github.com/google-research/met
cd met
pip install -r requirements.txt
```

## Standard Training (MET-S)

To train the MET-S model mentioned in the paper (model without adversarial training step) for FashionMNIST dataset, run this command:

```train
python3 train.py
```

The following hyper-parameters are available for train.py :
+ **embed_dim** : Embedding dimension
+ **ff_dim** : Feed-Forward dimension
+ **num_heads** : Number of heads
+ **model_depth_enc** : Depth of Encoder/ Number of transformers in Encoder stack
+ **model_depth_dec** : Depth of Decoder/ Number of transformers in Decoder stack
+ **mask_pct** : Masking Percentage
+ **lr** : Learning rate

Each of the above can be changed by adding --flag_name=flag_value to train.py. For example :
```
python3 train.py --model_depth_enc=1
```

The model is saved [here](./saved_models/) by default

## Adversarial Training (MET)

To train the MET model in the paper for FashionMNIST dataset trained using Adversarial training, run this command:
```train
python3 train_adv.py
```

The following hyper-parameters are available for train.py :
+ **embed_dim** : Embedding dimension
+ **ff_dim** : Feed-Forward dimension
+ **num_heads** : Number of heads
+ **model_depth_enc** : Depth of Encoder/ Number of transformers in Encoder stack
+ **model_depth_dec** : Depth of Decoder/ Number of transformers in Decoder stack
+ **mask_pct** : Masking Percentage
+ **lr** : Learning rate
+ **radius** : Radius of L2 norm ball around the input data point
+ **adv_steps** : Adversarial loop length
+ **lr_adv** : Adversarial Learning Rate

Each of the above can be changed by adding --flag_name=flag_value to train.py. For example :
```
python3 train_adv.py --radius=14
```

The model is saved [here](./saved_models/) by default

## Adding a new dataset :

You can try using the model on any new dataset by creating a csv file. The first column of the csv file should be class followed by the attributes. Sample csv files are available in [data](./data/)

To pass on the csv file to any of the training and evaluation scripts use the following flags :
+ **num_classes** : Number of classes
+ **model_kw** : Keyword for model (Eg fmnist for fashion-mnist)
+ **train_len** : Length of train csv
+ **train_data_path** : Path to train csv file
+ **test_len** : Length of test csv
+ **test_data_path** : Path to test csv files

- By default models are stored in [saved_models](./saved_models/). You can change the training path using flag **model_path**.
- Synthetic dataset can be created using [get_2d_dataset.py](./data/get_2d_data.py). By default a created dataset is available in [data](./data/2d_train.csv)

## Pre-trained Models

Pretrained models for FashionMNIST for optimal adversarial training setting is available in [saved_models](./saved_models/). You can extract the models using command:
```7z
7z e fmnist_saved.7z.001
```
```
7z e fmnist_saved_adv.7z.001
```

## Evaluation

To evaluate the saved MET-S model run
```eval
python3 eval.py --model_path="./saved_models/fmnist_64_1_64_6_1_70_1e-05" --model_path_linear="./saved_models/fmnist_linear_64_1_64_6_1_70_1e-05"
```

To evaluate the saved MET model run
```
python3 eval.py --model_path="./saved_models/fmnist_adv_64_1_64_6_1_70_1e-05" --model_path_linear="./saved_models/fmnist_linear_adv_64_1_64_6_1_70_1e-05"
```

By default results are written to **met.csv**.

## Results

### The performance of our model across various multi-class classification datasets is shown below.
<br>
<table>
<thead>
  <tr>
    <th>Type</th>
    <th>Methods</th>
    <th>FMNIST</th>
    <th>CIFAR10</th>
    <th>MNIST</th>
    <th>CovType</th>
    <th>Income</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">Supervised Baseline</td>
    <td>MLP</td>
    <td>87.57 &pm; 0.13</td>
    <td>16.47 &pm; 0.23</td>
    <td>96.98 &pm; 0.1</td>
    <td>65.45 &pm; 0.09</td>
    <td>84.35 &pm; 0.11</td>
  </tr>
  <tr>
    <td>RF</td>
    <td>87.19 &pm; 0.09</td>
    <td>36.75 &pm; 0.17</td>
    <td>97.62 &pm; 0.18</td>
    <td>64.94 &pm; 0.12</td>
    <td>84.6 &pm; 0.2</td>
  </tr>
  <tr>
    <td>GBDT</td>
    <td>88.71 &pm; 0.07</td>
    <td>45.7 &pm;&nbsp;&nbsp;0.27</td>
    <td>100 &pm; 0.0</td>
    <td>72.96 &pm; 0.11</td>
    <td>86.01 &pm; 0.06</td>
  </tr>
  <tr>
    <td>RF-G</td>
    <td>89.84 &pm; 0.08</td>
    <td>29.28 &pm; 0.16</td>
    <td>97.63 &pm; 0.03</td>
    <td>71.53 &pm; 0.06</td>
    <td>85.57 &pm; 0.13</td>
  </tr>
  <tr>
    <td>MET-R</td>
    <td>88.81 &pm; 0.12</td>
    <td>28.97 &pm; 0.08</td>
    <td>97.43 &pm; 0.02</td>
    <td>69.68 &pm; 0.07</td>
    <td>75.50 &pm; 0.04</td>
  </tr>
  <tr>
    <td rowspan="3">Self-Supervised Methods</td>
    <td>VIME</td>
    <td>80.36 &pm; 0.02</td>
    <td>34 &pm; 0.5</td>
    <td>95.74 &pm; 0.03</td>
    <td>62.78 &pm; 0.02</td>
    <td>85.99 &pm; 0.04</td>
  </tr>
  <tr>
    <td>DACL+</td>
    <td>81.38 &pm; 0.03</td>
    <td>39.7 &pm; 0.06</td>
    <td>91.35 &pm; 0.075</td>
    <td>64.17 &pm; 0.12</td>
    <td>84.46 &pm; 0.03</td>
  </tr>
  <tr>
    <td>SubTab</td>
    <td>87.58 &pm; 0.03</td>
    <td>39.32 &pm; 0.04</td>
    <td>98.31 &pm; 0.06</td>
    <td>42.36 &pm; 0.03</td>
    <td>84.41 &pm; 0.06</td>
  </tr>
  <tr>
    <td rowspan="2">Our Method</td>
    <td>MET-S</td>
    <td>90.90 &pm; 0.06</td>
    <td>47.96&nbsp;&nbsp;&pm;&nbsp;&nbsp;0.1</td>
    <td>98.98 &pm; 0.05</td>
    <td>74.13 &pm; 0.04</td>
    <td>86.17&nbsp;&nbsp;&pm;&nbsp;&nbsp;0.08</td>
  </tr>
  <tr>
    <td>MET</td>
    <td>91.68 &pm; 0.12</td>
    <td>47.92&nbsp;&nbsp;&pm;&nbsp;&nbsp;0.13</td>
    <td>99.17+-0.04</td>
    <td>76.68&nbsp;&nbsp;&pm;&nbsp;&nbsp;0.12</td>
    <td>86.21 &pm; 0.05</td>
  </tr>
</tbody>
</table>

### The performance of our model across various binary classification datasets is shown below.
<br>
<table>
<thead>
  <tr>
    <th>Datasets</th>
    <th>Metric</th>
    <th>MLP</th>
    <th>RF</th>
    <th>GBDT</th>
    <th>RF-G</th>
    <th>MET-R</th>
    <th>DACL+</th>
    <th>VIME</th>
    <th>SubTab</th>
    <th>MET</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Obesity</td>
    <td>Accuracy</td>
    <td>58.1 &pm; 0.07</td>
    <td>65.99 &pm; 0.12</td>
    <td>67.19 &pm; 0.04</td>
    <td>58.39 &pm; 0.17</td>
    <td>58.8 &pm; 0.59</td>
    <td>62.34 &pm; 0.12</td>
    <td>59.23 &pm; 0.17</td>
    <td>67.48 &pm; 0.03</td>
    <td>74.38 &pm; 0.13</td>
  </tr>
  <tr>
    <td>AUROC</td>
    <td>52.3 &pm; 0.12</td>
    <td>64.36 &pm; 0.07</td>
    <td>64.4 &pm; 0.05</td>
    <td>54.45 &pm; 0.08</td>
    <td>53.2 &pm; 0.18</td>
    <td>61.18 &pm; 0.07</td>
    <td>57.27 &pm; 0.21</td>
    <td>64.92 &pm; 0.06</td>
    <td>71.84 &pm; 0.15</td>
  </tr>
  <tr>
    <td rowspan="2">Income</td>
    <td>Accuracy</td>
    <td>84.35 &pm; 0.11</td>
    <td>84.6 &pm; 0.2</td>
    <td>86.01 &pm; 0.06</td>
    <td>85.57 &pm; 0.13</td>
    <td>75.50 &pm; 0.04</td>
    <td>85.99 &pm; 0.24</td>
    <td>84.46 &pm; 0.03</td>
    <td>84.41 &pm; 0.06</td>
    <td>86.21 &pm; 0.05</td>
  </tr>
  <tr>
    <td>AUROC</td>
    <td>89.39 &pm; 0.2</td>
    <td>91.53 &pm; 0.32</td>
    <td>92.5 &pm; 0.08</td>
    <td>90.09 &pm; 0.57</td>
    <td>83.48 &pm; 0.23</td>
    <td>89.01 &pm; 0.4</td>
    <td>87.37 &pm; 0.07</td>
    <td>88.95 &pm; 0.19</td>
    <td>93.85 &pm; 0.33</td>
  </tr>
  <tr>
    <td rowspan="2">Criteo</td>
    <td>Accuracy</td>
    <td>74.28 &pm; 0.32</td>
    <td>71.09 &pm; 0.05</td>
    <td>72.03 &pm; 0.03</td>
    <td>74.62 &pm; 0.08</td>
    <td>73.57 &pm; 0.12</td>
    <td>69.82 &pm; 0.06</td>
    <td>68.78 &pm; 0.13</td>
    <td>73.02 &pm; 0.08</td>
    <td>78.49 &pm; 0.05</td>
  </tr>
  <tr>
    <td>AUROC</td>
    <td>79.82 &pm; 0.17</td>
    <td>77.57 &pm; 0.1</td>
    <td>78.77 &pm; 0.04</td>
    <td>80.32 &pm; 0.16</td>
    <td>79.17 &pm; 0.17</td>
    <td>75.32 &pm; 0.27</td>
    <td>74.28 &pm; 0.39</td>
    <td>76.57 &pm; 0.05</td>
    <td>86.17 &pm; 0.2</td>
  </tr>
  <tr>
    <td rowspan="2">Arrhythmia</td>
    <td>Accuracy</td>
    <td>59.7 &pm; 0.02</td>
    <td>68.18 &pm; 0.02</td>
    <td>69.79 &pm; 0.12</td>
    <td>60.6 &pm; 0.05</td>
    <td>51.67 &pm; 0.1</td>
    <td>57.81 &pm; 0.47</td>
    <td>56.06 &pm; 0.04</td>
    <td>60.1 &pm; 0.1</td>
    <td>81.25 &pm; 0.12</td>
  </tr>
  <tr>
    <td>AUROC</td>
    <td>72.23 &pm; 0.06</td>
    <td>90.63 &pm; 0.08</td>
    <td>92.19 &pm; 0.05</td>
    <td>74.02 &pm; 0.12</td>
    <td>58.36 &pm; 0.32</td>
    <td>69.23 &pm; 0.98</td>
    <td>67.03 &pm; 0.27</td>
    <td>69.97 &pm; 0.07</td>
    <td>98.75 &pm; 0.04</td>
  </tr>
  <tr>
    <td rowspan="2">Thyroid</td>
    <td>Accuracy</td>
    <td>50 &pm; 0.0</td>
    <td>94.94 &pm; 0.1</td>
    <td>96.44 &pm; 0.07</td>
    <td>50 &pm; 0.0</td>
    <td>57.42 &pm; 0.37</td>
    <td>60.03 &pm; 0.05</td>
    <td>66.1 &pm; 0.19</td>
    <td>59.9 &pm; 0.16</td>
    <td>98.1 &pm; 0.08</td>
  </tr>
  <tr>
    <td>AUROC</td>
    <td>62.3 &pm; 0.12</td>
    <td>99.62 &pm; 0.03</td>
    <td>99.34 &pm; 0.02</td>
    <td>52.65 &pm; 0.13</td>
    <td>82.03 &pm; 0.26</td>
    <td>86.63 &pm; 0.1</td>
    <td>94.87 &pm; 0.03</td>
    <td>88.93 &pm; 0.12</td>
    <td>99.81 &pm; 0.09</td>
  </tr>
</tbody>
</table>