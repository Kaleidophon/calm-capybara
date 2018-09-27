## Comparing different Neural NLP models for the SemEval 2018 Task 2: Multilingual Emoji Prediction

This project was conducted for the *Deep Learning for Natural Language Technologies* course of the Universiteit van Amsterdam's
Master Artificial Intelligence program during the winter term 2018/2019. 

### Task description

This SemEval shared tasks aims to explore the predictability of emojis, small ideograms depicting objects, people and
scenes, given the tweet they were used in. An example tweet from the data set is the following:

    Addicted to #avocado toast. @ Kitchen :heart_eyes:
    
Only tweets with one emoji were included in the data set. No meta data is given. Because only tweets with one of the 20 
most frequent emojis in English or Spanish respectively were selected, this task can be seen as a form of multi-label 
classification. The tweets were gathered between October 2015 and February 2017. For more information 
about the task, consult the original [competition paper](http://www.aclweb.org/anthology/S18-1003).

### Results

The results are split up for the english and the spanish part of the data set. All models' performances were determined 
on the test set. Due to limited computational resources, the vocabulary size during training 
was limited to 10.000 types. Additionally, the data was pre-processed with the [Ekphrasis](https://github.com/cbaziotis/ekphrasis)
library, which supplies Twitter-specific tools for text normalization. 

#### English 

|  Model|Precision |Recall  |F1-score  |
|------:|:----------|:-------|:---------|
|Bag-of-Words + Logistic Regression | 0.3450 | 0.3069 | 0.3129 |
|Bag-of-Words + SVM  | 0.2586 | 0.2681 | 0.2549 |
| CNN | 0.3770 | 0.2809 | 0.2798 |

#### Spanish

| Model | Precision | Recall | F1-score |
|------:|:----------|:-------|:---------|


### Usage

1. Install the dependencies with

```sh
pip install -r requirements.txt
```

2. Read the datasets, process them and serialize the results:

```sh
python tweet_data.py
```

3. Read pretrained embeddings and serialize only those in the vocabulary:

```sh
python embeddings.py
```

4. Train a simple LSTM classifier:

```sh
python lstm.py
```

5. Monitor training with Tensorboard by running (in the same directory)

```sh
tensorboard --logdir=runs
```

and going to [http://localhost:6006](http://localhost:6006)
