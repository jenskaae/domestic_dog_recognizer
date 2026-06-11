# Domestic Dog Recognizer
**Question:** Is there a domestic dog in the picture?
#### Description
Binary image classifier for detecting whether an image contains a domestic dog or not.

Built with PyTorch and ResNet18 transfer learning.

The primary focus of the project is data quality and target definition rather than maximizing accuracy.

#### Motivation

The original goal was to build a practical dog vs not_dog classifier.

During development it became apparent that several source datasets labelled as "dog" contained non-domestic canids such as:

- dhole
- dingo
- African hunting dog

This created a data-quality problem:

Should these animals be considered dogs or not?

Because the project target is domestic dog recognition, these categories were later recategorized as not_dog.

The project therefore became a practical example of how dataset labels and project labels are not always the same thing.

#### Model Choice

The project uses a ResNet18 convolutional neural network with pretrained ImageNet weights.

ResNet18 was selected because it provides a good balance between model size, training speed and classification performance. The objective of the project was not to achieve state-of-the-art image recognition results, but to build a complete and reproducible machine-learning pipeline covering:

- data collection
    
- manifest generation
    
- data cleaning
    
- deduplication
    
- target-definition analysis
    
- model training
    
- validation diagnostics
    
- error analysis
    

Transfer learning allowed the model to benefit from features learned on ImageNet while requiring only a relatively small amount of project-specific training.

Only the final classification layer was replaced and trained for the binary classification task:

- domestic dog
    
- not_dog
    

Training was performed on CPU.

#### Data Sources

This project builds upon publicly available dog image datasets and additional manually collected negative examples.

##### Stanford Dogs Dataset

The Stanford Dogs Dataset provides images of many domestic dog breeds and was used as one of the primary positive-image sources. http://vision.stanford.edu/aditya86/ImageNetDogs/?utm_source=chatgpt.com the set contains about 120 breeds with around 150 images per breed.

##### Dog Breeds Image Dataset

A second dog-breed dataset obtained through Kaggle was used to increase breed coverage and image diversity. Dog Breeds Image Dataset [Dataset]. Kaggle. [https://www.kaggle.com/datasets/darshanthakare/dog-breeds-image-dataset/](https://www.kaggle.com/datasets/darshanthakare/dog-breeds-image-dataset/)

##### Additional Not-Dog Images

To support binary classification, additional images representing non-dog categories were collected and added as negative examples. These include animals, people, vehicles, natural scenes and other objects that should not be classified as domestic dogs.

##### Why Multiple Sources?

Combining multiple datasets increased breed coverage and image diversity, but also introduced inconsistencies that later required investigation and cleaning.

## Target Definition and Non-Domestic Canids

One of the most interesting findings of the project emerged during dataset inspection and error analysis.

From the beginning, it seemed natural to include visually challenging negative examples such as:

- wolves
    
- foxes
    
- hyenas
    
- large cats
    

The motivation was to force the model to learn a stronger distinction between domestic dogs and similar-looking animals.

However, while investigating classification errors and breed distributions, it became apparent that some of the source dog datasets already contained non-domestic canids, including:

- dhole
    
- dingo
    
- African hunting dog
    

These animals belong to the wider canid family but are not domestic dogs.

This raised an important question:

**What exactly is the target class?**

The project goal was defined as domestic dog recognition rather than recognition of all canids. As a result, these categories were recategorized from `dog` to `not_dog`.

This decision reduced the apparent classification accuracy compared with earlier experiments, but it aligned the dataset more closely with the actual project objective.

The experience highlighted an important lesson in machine learning:

Dataset labels and project labels are not necessarily the same thing. A model can achieve high accuracy while still solving the wrong problem if the target definition is not examined carefully.

#### Table of contents:

1. Get dog images from known selections, i.e. Kaggle
2. Build_manifest.py
3. Hash_and_deduplicate.py
4. Reclassify_non_domestic_canids.py
5. Create_train_val_test_split.py
6. Create_balanced_training_manifest.py
7. Train_resnet18_transfer_model.py
8. Validation_diagnostics.py
9. Error_analysis.py


### 1. Get dog images from known selections, i.e. Kaggle
Downloading dog images from Kaggle.com requires no real explanation, but collecting a significant amount of dog-negatives can probaly be a little cumbersome. The Python package ddgs, available through PyPI, was used. It offers Duckduckgo-search or now ddgs, a package that allows you to search for pictures to download from various internet sites. This offers the possibility to reach a diverse set of sources to prevent any bias. An example of how it works can be seen in /scripts/download_images_ddgs.py, which is the exact script I used.

### 2. Build a manifest as saved dataframe
The taske here is to normalize breed names across dog datasets. in order to make it reasonably simple to sort among breeds and make sure that all images concerning the same breed can be filtered in as simple a way possible. 
 

