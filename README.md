# Project 1

Report content (10):

## Data description and preprocessing 
> (if necessary normalization, feature selection,
transformation, etc.). Motivation for choosing the particular problem.

Because of the nature of our problem (image classification) we need to aprouch this with that in mind 

- normalization/feature selection are kind of 

## Data visualization
> (histograms, box plots, other plots).

Imagens msm com as labels
 

## Short description of the implemented ML models.

-> Logistic Regression I guess
-> MLP feed forward nn

## Model training 
> (data splitting – train, validate, test, k-fold Cross validation). Visualize
graphically the cost function trajectory over iterations. Training with regularized and nonregularized cost function.

show cost function tajectory and accuracy

##  Model hyper-parameter selection 
> regularization parameter lambda, number of NN hidden
layer units, number of hidden layers (if necessary), sigma, C, k, etc.. Systematic approach
instead of just one or several randomly chosen values.

get the hyoer parammeters -> greed search cv


## For a classification problem, you need to present the confusion Matrix 
> (accuracy, precision,
recall, F1 score, etc.).

        apple   axe     book    house   sword

apple   28078   320     215     128     131
axe     122     23692   315     140     521
book    181     522     22846   203     115
house   327     353     677     25550   121
sword   110     784     123     120     23792

## Performance comparison between the models.

- logistic regression 
- MLP nn

## Results in graphical or table formats.

- Grafico de evolucao de accuracy 
- Neural Network accuracy: 0.9737770794157801 for neurons = (784,300,60,50,40,10), 784 é bom por causa de ser 28*28 qualquer outro numero vai ser pior

## Conclusions.

## Problem complexity. 

iamgens n podemos k fold crossvalidation

divide train, validacao, teste


# ToDos


# Notas

- Correct data spliting stupid Implementation