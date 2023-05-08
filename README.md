# Project 1

Report content (10):

## Data description and preprocessing 
> (if necessary normalization, feature selection,
transformation, etc.). Motivation for choosing the particular problem.

Because of the nature of our problem (image classification) we need to aprouch this with that in mind 

- We choose to use normalization : Normalizing the pixel values of the images can help improve convergence during training. This typically involves scaling the pixel values to a specific range, such as [0, 1] or [-1, 1]. we did [0, 1]

thats it 

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


Using a form of validation:

Best validation accuracy: 0.9534930417187959 
Best learning rate: 0.001
Best momentum coefficient: 0.99


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

 
Apple 
        Precision:
        Recall:
        F1 Score: 

axe  
        Precision:
        Recall:
        F1 Score: 
book 
        Precision: 0.94498676373 = 22846 / (215 + 315 + 22846 + 677 + 123)
        Recall: 0.95722126785 = 22846 /(181 + 522 + 22846 + 203 + 115)
        F1 Score: 

house 
        Precision: 0.97739183657 = 25550 / (128 + 140 + 203 + 25550 + 120)
        Recall: 0.94531596862 = 25550 / (327 + 353 + 677 + 25550 + 121)
        F1 Score: 

sword 
        Precision: 0.96401944894 = 23792 / (131 + 521 + 115 + 121 + 23792)
        Recall: 0.95439046893 = 23792 / (110 + 784 + 123 + 120 + 23792)
        F1 Score: 

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