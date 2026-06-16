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

#### Repo structure:
## Repository Structure

domestic_dog_recognizer/
├── data/
├── models/
├── scripts/
├── README.md
└── requirements.txt

| Folder           | Purpose                                                    |
| ---------------- | ---------------------------------------------------------- |
| data/            | Generated manifests, diagnostics and intermediate datasets |
| models/          | Trained model weights                                      |
| scripts/         | Pipeline scripts used throughout the project               |
| README.md        | Project description and methodology                        |
| requirements.txt | Python dependencies                                        |

#### Data Sources

This project builds upon publicly available dog image datasets and additional manually collected negative examples.

##### Stanford Dogs Dataset

The Stanford Dogs Dataset provides images of many domestic dog breeds and was used as one of the primary positive-image sources. http://vision.stanford.edu/aditya86/ImageNetDogs the set contains about 120 breeds with around 150 images per breed.

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
2. Normalize data and build manifest
3. Hashes, label analysis and deduplication
4. Reclassify non-domestic canids.py
5. Create train val test split
6. Create balanced training manifest
7. Train_resnet18_transfer_model.py
8. Validation_diagnostics.py
9. Error_analysis.py


### 1. Get dog images from known selections, i.e. Kaggle

Downloading dog images from Kaggle.com requires no real explanation, but collecting a significant amount of dog-negatives can probaly be a little cumbersome. The Python package ddgs, available through PyPI, was used. It offers Duckduckgo-search or now ddgs, a package that allows you to search for pictures to download from various internet sites. This offers the possibility to reach a diverse set of sources to prevent any bias. An example of how it works can be seen in /scripts/download_images_ddgs.py, which is the exact script I used for not_dog image collection.

### 2. Normalize data and build manifest

The task here is to normalize breed names across dog datasets. In the dog breeds dataset the breed name is the folders name that holds the collection of images of the breed, whereas in the Stanford set, the breed name was only a part of the assembly folders name, also the name was spelled differently across datasets, to give a few examples:

Stanford  Dogs Dataset                     Dog Breeds Dataset     Normalized
n02116738-African_hunting_dog      african                          african_hunting_dog
n02113978-Mexican_hairless             mexicanhairless           mexican_hairless


in order to make it reasonably simple to sort and group by breeds and make sure that all images concerning the same breed can be filtered in as simple a way possible, it was necessary to normalize the breed names, and in some cases even check whether the names actually point to the same breed as in the example african vs. African hunting dog.

At first I tried matching candidates by simple similarity score, however, whichever score threshold I chose there would be far to many false positives or negatives to make it a simple task to find all candidates, I thus had to develop the more rigorous tokenization and token-matching method.

The work to sort this out was done using pandas dataframes, a few sample diagnostic and action scripts can be seen in /scripts/normalize_data_and_build_manifest.py
 
### 3. Hashes, label analysis and deduplication

The two dog datasets were collected independently, so it could not be assumed that the images were unique across sources. Before training, I therefore checked whether identical images appeared in both datasets.

Several hash types were calculated for each image. Cryptographic hashes identify byte-identical files, while perceptual hashes provide a basis for later image-similarity investigations.

|Hash|Purpose|
|---|---|
|MD5|Fast exact file comparison|
|SHA256|Robust exact file comparison|
|aHash|Perceptual similarity|
|dHash|Perceptual similarity|
|pHash|Perceptual similarity|

The analysis produced the following results:

- 39,045 images in total
- 16,874 duplicate images
- 22,171 unique images

The number of duplicates was much higher than expected. The merged dataset contained approximately 5,000 additional unique images compared to the larger source dataset alone. However, these additional samples were almost entirely dog images and therefore did not address the substantial class imbalance between dog and not_dog categories.

The duplicate analysis revealed a small number of images that had been assigned different breed labels across the two datasets. Since it was not possible to determine the correct breed automatically, all images involved in such label conflicts were removed entirely.

The final result was a deduplicated manifest containing only unique images and label-consistent samples for subsequent training. Code used to achieve some of these goals are represented in /scripts/hashes_label_analyses_and_deduplication.py

### 4. Reclassify non-domestic canids

The initial project goal was to distinguish dogs from non-dogs.

A first training run using the original labels produced a high-performing classifier. However, inspection of the validation mistakes revealed an interesting pattern: several of the most difficult cases involved canids that are biologically related to domestic dogs.

At the start of the project I had manually classified wolves, foxes and similar animals as not_dog. Later analysis revealed that the two source dog datasets also contained other non-domestic canids, including dholes, dingoes and African hunting dogs, but these had been labelled as dog.

This raised an important question:

> What exactly is a dog?

Before any reclassification  the number of dogs (1) and not-dogs(0) were:
class   count
1       38078 
0           967

after dedup:

class   count
1       20857 
0           966 


and the number of dholes, African hunting dogs and dingos were:

normalized_breed 
african_hunting_dog     338 
dingo                             304 
dhole                             300

After training I profiled the set of false negatives, i.e., dogs categorized as not-dogs:

Among the false negatives, dholes and African hunting dogs appeared disproportionately often:


normalized_breed 
dhole                               15 
komondor                         9 
african_hunting_dog         6 
chihuahua                          3 
chesapeake_bay_retriever  2 
irish_water_spaniel              2 
weimaraner                         2 
vizsla                                   2 
afghan_hound                     1 
airedale                                1 
australian_cattledog             1 
chow                                     1 
basenji                                  1 
bloodhound                          1

Analysis of the validation mistakes showed that dholes and African hunting dogs were among the most frequently misclassified positive example

Apart from the komondor, which may be mistaken by some as a floor mob and that likely explains the struggle of the model in the komondor case, it is found interesting that dholes and African hunting dogs would not normally be classified as dog, but still are members of the canid family of dog-like animals which counts species like wolf, fox and other animals that looks like dogs. 

From a biological perspective, domestic dogs are members of the canid family and do not form a clearly separate visual category from all other canids. As a result, the training data contained contradictory signals:

- some canids were labelled dog (dhole, African hunting dog, dingo ....),
- other canids were labelled not_dog (wolf, fox ....).

The training data therefore contained contradictory signals. Wolves and foxes were explicitly labelled as not_dog, while other non-domestic canids such as dholes, dingoes and African hunting dogs were labelled as dog. The model was therefore asked to learn a decision boundary that was inconsistent with the intended project definition.

To align the dataset with this objective, known non-domestic canids were reclassified from dog to not_dog before retraining.

This was less an attempt to improve accuracy and more an attempt to improve the consistency of the target definition presented to the model:

Before recategorization, the dataset contained 942 images of African hunting dogs, dingoes and dholes. After deduplication, 468 of these images remained in the training population.

After recategorization, 
all rows: 
class_label 
1 37136 
0 1909 

After recategorization, keep=True: (keep = true => deduplicated)
class_label 
1 20389 
0 1434 

Rows recategorized, 
all rows: 942 
Rows recategorized, keep=True: 468 (keep = true => deduplicated)

Reclassified canids remaining after deduplication: 
normalized_breed class_label 
african_hunting_dog 0 169 
dingo 0 155 
dhole 0 144 

After deduplication, 468 retained images were reclassified from dog to not_dog. The purpose was not primarily to improve accuracy, but to ensure that the target labels reflected the intended task: recognizing domestic dogs rather than canids in general.

The diagnostics scripts are found in /scripts/reclassify_non_domestic_canids.py

### 5. Create train-val-test split

After breed normalization, duplicate handling, and reclassification of non-domestic canids, the cleaned manifest was split into training, validation, and test sets.

Only rows marked with `keep == True` were used for the split.

Final split sizes:
class_label not_dog    dog 
split 
test                  209   3054 
train               1009 14289 
val                    216   3046 

Final split percentages:
class_label not_dog    dog 
split 
test                 0.064 0.936 
train                0.066 0.934 
val                   0.066 0.934        almost equal distributions of dog/not dog in sets

sum: 

| Split      | Images |
| ---------- | -----: |
| Train      | 15,298 |
| Validation |  3,262 |
| Test       |  3,263 |

The training set was intentionally kept as the largest subset, while validation and test were kept separate. The validation set was used during model development and diagnostics, while the test set was reserved as a final holdout set.

The split was performed after reclassifying non-domestic canids, so the dog / not-dog labels reflect the final project definition:

> domestic dog vs. not domestic dog.

The training set still remained highly imbalanced before balancing:

|Class|Training images|
|---|--:|
|Dog|14,289|
|Not dog|1,009|

This imbalance is handled in the next step by creating a reduced and more balanced training manifest.

The splitting scripts are found in /scripts/create_train_val_test_split.py

### 6. Balanced training manifest

The original training split was highly imbalanced:

|Class|Images|
|---|--:|
|Dog|14,289|
|Not dog|1,009|

Training directly on this distribution would encourage the model to favor the majority class.

To reduce this imbalance, all not-dog images were retained while the number of dog images was limited on a per-breed basis.

A maximum of seven images per normalized breed was sampled:

```python
MAX_DOGS_PER_BREED = 7
```

Breeds containing more than seven training images were randomly reduced to seven samples. Breeds containing seven or fewer training images were retained in full.

This approach preserves breed diversity while preventing a small number of common breeds from dominating the training set.

Seven images per breed was chosen empirically because it produced a training set of roughly the same size as the not-dog class while still retaining representation from all available breeds.

The resulting balanced training manifest contained:

|Class|Images|
|---|--:|
|Dog|965|
|Not dog|1,009|

A total of 156 normalized dog breeds remained represented in the balanced dataset.

The objective was not to create perfect class equality, but to reduce the extreme imbalance while maintaining broad coverage of domestic dog appearances.

Validation and test sets were left unchanged so evaluation continued to reflect the natural distribution of the cleaned dataset.

### 7. Train resnet18 transfer model

The reduced training manifest from the previous step is used to train a binary image classifier:

- `1` = domestic dog
- `0` = not domestic dog

The validation and test sets are not reduced. Only the training set is balanced. This keeps evaluation closer to the real distribution of the prepared dataset while reducing the risk that the model simply learns to predict the majority class during training.

The model uses transfer learning with `ResNet18` pretrained on ImageNet. The pretrained convolutional backbone is frozen, and only the final classification layer is replaced and trained for this project.

The training script performs the following steps:

1. Load `manifest_train_reduced.csv` and `manifest_val.csv`
2. Create a custom PyTorch `Dataset` from the manifest rows
3. Use the default preprocessing transforms expected by pretrained ResNet18
4. Create PyTorch `DataLoader`s for training and validation
5. Replace the final ResNet18 layer with a two-class classifier
6. Train the final layer for a small number of epochs
7. Save the trained model weights

The purpose of this step is not to build a state-of-the-art classifier. The purpose is to establish a clean baseline model after the dataset has been deduplicated, recategorized, split, and balanced.

Example output:

```
Train labels:
0    1009
1     965

Validation labels:
1    dog
0    not_dog

Using device: cpu

Epoch 1/5 | Loss: 17.6401 | Train acc: 0.927 | Val acc: 0.985  
Epoch 2/5 | Loss: 9.0755 | Train acc: 0.968 | Val acc: 0.970
Epoch 3/5 | Loss: 7.3681 | Train acc: 0.972 | Val acc: 0.968
Epoch 4/5 | Loss: 6.2760 | Train acc: 0.977 | Val acc: 0.968
Epoch 5/5 | Loss: 5.9290 | Train acc: 0.975 | Val acc: 0.979 
Saved model to: models/resnet18_dog_not_dog.pt
```


### 8. Validation diagnostics

After training the baseline ResNet18 model, the next step is to inspect its validation predictions in more detail.

The training script reports aggregate accuracy, but accuracy alone does not explain what kind of mistakes the model makes. For this project, the mistakes are important because the central question is whether the model has learned the intended target:

> domestic dog vs not domestic dog

The validation diagnostics script performs the following steps:

1. Load the saved ResNet18 model weights
2. Load the validation manifest
3. Run the model on all validation images
4. Store the predicted class, actual class, confidence score, and image path
5. Merge the prediction results with the original manifest metadata
6. Save a full diagnostics file
7. Save a separate file containing only mistakes

The output files are:

data/intermittent/validation_diagnostics.csv
data/intermittent/validation_mistakes.csv

A look on the quick results list, most presented false negative breeds in particular, reveals a pattern:

Using device: cpu 
Validation accuracy: 0.974
Validation rows: 3262
Mistakes: 84 
Saved diagnostics: [C:\AI_Projects\domestic_dog_recognizer\data\intermittent\validation_diagnostics.csv](file:///C:/AI_Projects/domestic_dog_recognizer/data/intermittent/validation_diagnostics.csv) 
Saved mistakes: [C:\AI_Projects\domestic_dog_recognizer\data\intermittent\validation_mistakes.csv](file:///C:/AI_Projects/domestic_dog_recognizer/data/intermittent/validation_mistakes.csv) 
Mistakes by actual/predicted: 
predicted 0    1 
actual
0               0 14 
1             70   0 

Top mistake breeds: 
normalized_breed 
dingo 5 
saluki 5 
mexican_hairless 5 
ibizan_hound 4 
whippet 4 
chihuahua 4 
german_shepherd 4 
cardigan_corgi 3 
afghan_hound 3 
rhodesian_ridgeback 3 
siberian_husky 3 
shiba 2 
malinois 2
weimaraner 2 
pembroke 2 
eskimo_dog 2 
newfoundland 1 
samoyed 1 
norfolk_terrier 1 
scottish_deerhound 1 

Most classification errors occurred among primitive, rare, or visually unusual dog breeds, together with biologically similar canids such as dingoes. This suggests that the remaining errors are concentrated near the project's target-definition boundary rather than among obviously unrelated categories.

##### Validation Metrics

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 97.4% |
| Precision | 99.5% |
| Recall    | 97.7% |
| F1 Score  | 98.6% |

The model achieved high precision and recall on the validation set. False positives were rare, indicating that the classifier was highly selective when predicting the domestic_dog class. Most remaining errors were false negatives concentrated among visually unusual breeds and biologically similar canids.

### 9. Error analysis

#### Highest-confidence false negatives

The most confident false negatives included:

- eskimo_dog
    
- whippet
    
- leonberg
    
- saluki
    
- norwegian_elkhound
    
- shiba
    
- ibizan_hound
    
- mexican_hairless
    

Several of these breeds differ substantially from the stereotypical appearance of a domestic dog. Common characteristics include primitive morphology, unusual body proportions, sparse coats, or visual similarity to wild canids.

#### False negatives by breed

The most frequently misclassified breeds were:

- saluki
    
- mexican_hairless
    
- whippet
    
- chihuahua
    
- ibizan_hound
    
- german_shepherd
    

Errors were concentrated among a relatively small number of breeds rather than being uniformly distributed across all breeds.

#### False positives

False positives were relatively uncommon.

The most frequent false positives included:

- dingo
    
- goat
    
- horse
    
- african_hunting_dog
    
- cougar
    

The dingo-related errors are particularly interesting because dingoes are visually similar to domestic dogs and were deliberately reclassified as not_dog during dataset preparation.

#### Discussion

The model rarely mistakes non-dogs for dogs. Most remaining errors occur when the image actually contains a dog but the dog's appearance differs substantially from common domestic dog breeds.

This supports the central observation of the project: defining the target class is often more important than optimizing model architecture. Much of the remaining error occurs near the boundary between domestic dogs and visually similar non-domestic canids.### Future work

A natural next step would be to treat non-domestic canids as deliberate hard negatives rather than ordinary not_dog examples. This could include wolves, coyotes, jackals, dholes, dingoes, foxes, and African wild dogs.

Because these animals share many visual traits with domestic dogs, the model would need to learn more subtle distinctions than those required for a simple canid vs non-canid classifier.

Possible improvements include:

- adding more hard-negative canid examples
- separating the task into three classes: `domestic_dog`, `non_domestic_canid`, and `other`
- comparing errors across primitive breeds, sled dogs, and wild canids
- training with stronger augmentation and/or a larger backbone
- fine-tuning more layers of the pretrained model instead of only the final layer
- using embedding analysis to inspect whether domestic dogs and wild canids form separable clusters
- testing whether confidence scores are lower around biologically ambiguous cases

The central challenge is that the project-specific target definition cuts across biological similarity. Improving the model therefore means helping it learn a boundary that is meaningful for the application, but not fully aligned with natural taxonomy.