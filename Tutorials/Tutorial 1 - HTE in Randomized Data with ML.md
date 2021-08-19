---
title: "Tutorial 1 - Estimating Heterogeneous Treatment Effects in Randomized Data with Machine Learning Techniques"
author: "Victor Hugo C. Alexandrino da Silva"
date: "8/11/2021"
output:
  html_document:
    df_print: paged
    keep_md: yes
  pdf_document: default
---




\section{1. Goal}

This tutorial discusses and implements different methods to estimate heterogeneous treatment effects (HTE) in \textbf{randomized data}:

\begin{itemize}
\item OLS with interaction terms
\item Post-selection LASSO
\item Honest Causal Tree
\item Causal Forest
\end{itemize}

We will compare the heterogeneity in each of these methods and then compare the \textbf{conditional average treatment effect (CATE)} in each of these methods.

Then, we compare the causal models by the mean square error (MSE).

\section{2. Set-up}

\subsection{2.1. Packages}

First, let's define our directory and install required packages:


```r
# Directory
setwd('~/Google Drive/PhD Insper/Thesis/Paper 3/Empirics/Tutorials/HTE Randomized')

# Packages
library(glmnet)               # LASSO
library(rpart)    
library(rpart.plot)
library(randomForest)         # Random Forest
library(devtools)             # GitHub installation
library(tidyverse)    
library(ggplot2)
library(dplyr)                # Data manipulation
library(grf)                  # Generalized Random Forests
#install_github('susanathey/causalTree')
library(causalTree)           # Causal Tree 
#install_github('swager/randomForestCI')
library(randomForestCI)
# install_github('swager/balanceHD')
library(balanceHD)            # Approximate residual balancing
library(SuperLearner)
library(caret)
library(xgboost)
library(sandwich)             # Robust SEs
library(ggthemes)             # Updated ggplot2 themes
library(iml)                  # For Shapley Values
```


\subsection{2.2. Data}

We used data from the article “Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment” by Gerber, Green and Larimer (2008). In a random experiment, the article study the effect of letters encouraging voters in the amount of voter turnout. The paper estimated that there is an average of 8 p.p. increase in turnout. 


Data is splitted in our set of variables $Z = \{Y_i,X_i,W_i\}$, where $Y$ is the outcome variable (voted or not), $X$ is the treatment variable (received letter or not) and $W$ the covariates. 


```r
# Load data
my_data <- readRDS('social_voting.rds')

#Restrict the sample size
n_obs <- 33000 # Change this number depending on the speed of your computer. 6000 is also fine. 
my_data <- my_data[sample(nrow(my_data), n_obs), ]

# Split data into 3 samples
folds = createFolds(1:nrow(my_data), k=2)

Y_train <- my_data[folds[[1]],1]
Y_test <- my_data[folds[[2]],1]

X_train <- my_data[folds[[1]],2]
X_test <- my_data[folds[[2]],2]

W_train <- my_data[folds[[1]],3:ncol(my_data)]
W_test <- my_data[folds[[2]],3:ncol(my_data)]

### Creates a vector of 0s and a vector of 1s of length n (hack for later usage)
zeros <- function(n) {
  return(integer(n))
}
ones <- function(n) {
  return(integer(n)+1)
}

summary(W_train)
```

```
##      g2000            g2002            p2000            p2002       
##  Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
##  1st Qu.:1.0000   1st Qu.:1.0000   1st Qu.:0.0000   1st Qu.:0.0000  
##  Median :1.0000   Median :1.0000   Median :0.0000   Median :0.0000  
##  Mean   :0.8665   Mean   :0.8382   Mean   :0.2659   Mean   :0.4092  
##  3rd Qu.:1.0000   3rd Qu.:1.0000   3rd Qu.:1.0000   3rd Qu.:1.0000  
##  Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
##      p2004             sex                 yob                 city         
##  Min.   :0.0000   Min.   :-1.007415   Min.   :-3.830877   Min.   :-0.55571  
##  1st Qu.:0.0000   1st Qu.:-1.007415   1st Qu.:-0.607789   1st Qu.:-0.51573  
##  Median :0.0000   Median : 0.992631   Median : 0.022815   Median :-0.47575  
##  Mean   :0.4204   Mean   : 0.003517   Mean   :-0.002103   Mean   :-0.01157  
##  3rd Qu.:1.0000   3rd Qu.: 0.992631   3rd Qu.: 0.653419   3rd Qu.:-0.11594  
##  Max.   :1.0000   Max.   : 0.992631   Max.   : 2.194896   Max.   : 3.44217  
##     hh_size          totalpopulation_estimate  percent_male     
##  Min.   :-1.263928   Min.   :-1.55267         Min.   :-4.31613  
##  1st Qu.:-1.263928   1st Qu.:-0.89431         1st Qu.:-0.57099  
##  Median : 0.121946   Median :-0.05730         Median :-0.11550  
##  Mean   :-0.008158   Mean   :-0.00244         Mean   :-0.00868  
##  3rd Qu.: 0.121946   3rd Qu.: 0.67929         3rd Qu.: 0.49182  
##  Max.   : 7.051318   Max.   : 3.31947         Max.   :13.80225  
##    median_age        percent_62yearsandover percent_white      
##  Min.   :-4.149665   Min.   :-2.856402      Min.   :-7.047567  
##  1st Qu.:-0.567389   1st Qu.:-0.659862      1st Qu.:-0.414015  
##  Median : 0.014731   Median :-0.099520      Median : 0.396475  
##  Mean   : 0.004668   Mean   : 0.007472      Mean   : 0.006835  
##  3rd Qu.: 0.574461   3rd Qu.: 0.483235      3rd Qu.: 0.658326  
##  Max.   : 4.671688   Max.   : 6.624583      Max.   : 0.932646  
##  percent_black       percent_asian       median_income      employ_20to64      
##  Min.   :-0.700360   Min.   :-0.686893   Min.   :-1.85360   Min.   :-8.349088  
##  1st Qu.:-0.553412   1st Qu.:-0.527512   1st Qu.:-0.71431   1st Qu.:-0.551770  
##  Median :-0.357482   Median :-0.341569   Median :-0.23317   Median : 0.192372  
##  Mean   :-0.009496   Mean   : 0.000774   Mean   : 0.00093   Mean   :-0.002386  
##  3rd Qu.: 0.132343   3rd Qu.: 0.056882   3rd Qu.: 0.49449   3rd Qu.: 0.645329  
##  Max.   : 9.781901   Max.   : 6.724294   Max.   : 4.36071   Max.   : 2.424800  
##    highschool        bach_orhigher       percent_hispanicorlatino
##  Min.   :-2.648769   Min.   :-1.848546   Min.   :-0.939459       
##  1st Qu.:-0.612953   1st Qu.:-0.784316   1st Qu.:-0.516044       
##  Median : 0.024637   Median :-0.116564   Median :-0.291882       
##  Mean   :-0.002946   Mean   : 0.003521   Mean   :-0.007839       
##  3rd Qu.: 0.762900   3rd Qu.: 0.509454   3rd Qu.: 0.081720       
##  Max.   : 2.821087   Max.   : 3.347401   Max.   : 8.425500
```


\section{3. CATE, Causal trees and causal forests}

\subsection{3.1. OLS with interaction terms}

We first estimate our CATE using the standard OLS. Let's start with a simple OLS with interaction terms. This is a simple way to estimate differential effects of $X$ on $Y$, where we only need to include interaction terms in an OLS regression. The algorithm follows:

\begin{itemize}
\item i) Regress $Y$ on $X$ and $W$
\item ii) Interact $X$ and $W$ in order to find heterogeneous effects of $X$ on $Y$ depending on $W$.  
\end{itemize}

Thus, we model an OLS model with interactions as the following:

$$ Y = \beta_0 + \beta_1 X + \beta_2 W + \beta_3 (W \times X) + \varepsilon $$
Where $X$ is the treatment vector and $W$ the covariates. 

We will use the R package `SuperLearner` to implement some of our ML algorithms:


```r
# Estimate a linear model algorithm
sl_lm = SuperLearner(Y = Y_train,
                     X = data.frame(X=X_train, W_train , W_train*X_train),
                     family = binomial(),              # Distribution of errors
                     SL.library = "SL.lm",             # Linear model
                     cvControl = list(V=0))            # Method for cross-validation

summary(sl_lm$fitLibrary$SL.lm_All$object)
```

```
## 
## Call:
## stats::lm(formula = Y ~ ., data = X, weights = obsWeights, model = model)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -0.7779 -0.3316 -0.2148  0.5330  0.9834 
## 
## Coefficients:
##                              Estimate Std. Error t value Pr(>|t|)    
## (Intercept)                 0.1370960  0.0131684  10.411  < 2e-16 ***
## X                           0.0894585  0.0321877   2.779 0.005454 ** 
## g2000                      -0.0414165  0.0122188  -3.390 0.000702 ***
## g2002                       0.0797954  0.0112986   7.062 1.70e-12 ***
## p2000                       0.1023194  0.0089014  11.495  < 2e-16 ***
## p2002                       0.1044448  0.0080530  12.970  < 2e-16 ***
## p2004                       0.1459407  0.0079066  18.458  < 2e-16 ***
## sex                        -0.0016974  0.0038194  -0.444 0.656745    
## yob                        -0.0253613  0.0042304  -5.995 2.08e-09 ***
## city                        0.0480217  0.0041187  11.659  < 2e-16 ***
## hh_size                     0.0150153  0.0040898   3.671 0.000242 ***
## totalpopulation_estimate    0.0005612  0.0049110   0.114 0.909029    
## percent_male               -0.0049598  0.0046328  -1.071 0.284378    
## median_age                  0.0065388  0.0093327   0.701 0.483540    
## percent_62yearsandover      0.0019783  0.0091538   0.216 0.828899    
## percent_white               0.0752785  0.0194075   3.879 0.000105 ***
## percent_black               0.0569334  0.0144737   3.934 8.40e-05 ***
## percent_asian               0.0294617  0.0102978   2.861 0.004229 ** 
## median_income               0.0213039  0.0087713   2.429 0.015158 *  
## employ_20to64              -0.0073377  0.0054333  -1.350 0.176874    
## highschool                  0.0452953  0.0138679   3.266 0.001092 ** 
## bach_orhigher               0.0219135  0.0152386   1.438 0.150446    
## percent_hispanicorlatino    0.0163558  0.0060430   2.707 0.006805 ** 
## g2000.1                    -0.0232479  0.0304228  -0.764 0.444783    
## g2002.1                     0.0207706  0.0276796   0.750 0.453029    
## p2000.1                    -0.0085342  0.0214774  -0.397 0.691108    
## p2002.1                    -0.0041624  0.0198668  -0.210 0.834049    
## p2004.1                     0.0195516  0.0194074   1.007 0.313742    
## sex.1                      -0.0127479  0.0093911  -1.357 0.174659    
## yob.1                      -0.0068105  0.0103044  -0.661 0.508666    
## city.1                      0.0072403  0.0103625   0.699 0.484752    
## hh_size.1                  -0.0153516  0.0097455  -1.575 0.115219    
## totalpopulation_estimate.1 -0.0109297  0.0124828  -0.876 0.381269    
## percent_male.1              0.0047269  0.0113512   0.416 0.677104    
## median_age.1               -0.0087187  0.0223506  -0.390 0.696476    
## percent_62yearsandover.1   -0.0092515  0.0218929  -0.423 0.672608    
## percent_white.1             0.0359065  0.0489502   0.734 0.463245    
## percent_black.1             0.0322912  0.0364996   0.885 0.376331    
## percent_asian.1             0.0213759  0.0257529   0.830 0.406529    
## median_income.1            -0.0007675  0.0211344  -0.036 0.971033    
## employ_20to64.1            -0.0167034  0.0131867  -1.267 0.205286    
## highschool.1               -0.0525802  0.0338497  -1.553 0.120360    
## bach_orhigher.1            -0.0616153  0.0370215  -1.664 0.096070 .  
## percent_hispanicorlatino.1 -0.0007316  0.0151743  -0.048 0.961545    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.4468 on 16456 degrees of freedom
## Multiple R-squared:  0.07547,	Adjusted R-squared:  0.07306 
## F-statistic: 31.24 on 43 and 16456 DF,  p-value: < 2.2e-16
```

Thus, note that the treatment $X$ itself has a positive statistically significant effect in the vote turnout. However, we observe statistically significant heterogeneous effect when we interact with some of the covariates. It seems that the effect is heterogeneous for our variables. Moreover, a bunch of features seems to be relevant for our linear model, which reinforces the need to estimate the CATE instead of the ATE. That is why we need one step forward.

\subsubsection{3.1.1. CATE for OLS}

We can simply predict our outcome for both treated and non-treated groups in order to estimate the CATE:

$$ CATE = \hat{\tau} = E(Y|X = 1,W) - E(Y|X=0,W) $$
For this, we use the `predict` function in our `sl_lm` model for each group, using our test sample:


```r
# Prediction on control group (X = 0)
ols_pred_control <- predict(sl_lm, data.frame(X = zeros(nrow(W_test)), W_test, W_test*zeros(nrow(W_test))), onlySL = T)

# Prediction on treated group (X = 1)
ols_pred_treated <- predict(sl_lm, data.frame(X = ones(nrow(W_test)), W_test, W_test*ones(nrow(W_test))), onlySL = T)

# Calculate CATE
cate_ols <- ols_pred_treated$pred - ols_pred_control$pred

plot(cate_ols)
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
# Calculate ATE
mean(cate_ols)
```

```
## [1] 0.09121995
```

```r
plot(mean(cate_ols))
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-4-2.png)<!-- -->


\subsection{3.2. Post-selection LASSO}

Now, we use LASSO to estimate the heterogeneous effects. However, before estimating the CATE, we use it as a screening algorithm. What does it mean? That in order to reduce the number of variables, we can use LASSO to select the relevant variables. We will use the SuperLearner library again:


```r
# Defining LASSO
lasso = create.Learner("SL.glmnet",
                       params = list(alpha = 1),
                       name_prefix = "lasso")


# Getting coefficients by LASSO
get_lasso_coeffs <- function(sl_lasso) {
  return(coef(sl_lasso$fitLibrary$lasso_1_All$object, se = "lambda.min")[-1,])
}


SL.library <- lasso$names

predict_y_lasso <- SuperLearner(Y = Y_train,
                                X = data.frame(X = X_train, W_train ,W_train*X_train),
                                family =  binomial(),
                                SL.library = SL.library,
                                cvControl = list(V=0))

kept_variables <- which(get_lasso_coeffs(predict_y_lasso)!=0)


predict_x_lasso <- SuperLearner(Y = X_train,
                                X = data.frame(W_train),
                                family = binomial(),
                                SL.library = lasso$names,
                                cvControl = list(V=0))


kept_variables2 <- which(get_lasso_coeffs(predict_x_lasso)!=0) + 1
```

After selecting the variables by LASSO, we can use the OLS to estimate treatment heterogeneity in the relevant variables selected by the algorithm above. But first, let's formulate our post selection OLS:


```r
sl_post_lasso <- SuperLearner(Y = Y_train,
                              X = data.frame(X = X_train, W_train, W_train*X_train)[,c(kept_variables,kept_variables2)],
                              family = binomial(),
                              SL.library = "SL.lm",
                              cvControl = list(V=0))

summary(sl_post_lasso$fitLibrary$SL.lm_All$object)
```

```
## 
## Call:
## stats::lm(formula = Y ~ ., data = X, weights = obsWeights, model = model)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -0.7076 -0.3344 -0.2270  0.5492  0.9707 
## 
## Coefficients:
##                Estimate Std. Error t value Pr(>|t|)    
## (Intercept)    0.116872   0.010447  11.187  < 2e-16 ***
## X              0.060656   0.024403   2.486 0.012942 *  
## g2002          0.064865   0.010800   6.006 1.94e-09 ***
## p2000          0.097591   0.008023  12.163  < 2e-16 ***
## p2002          0.103287   0.007335  14.081  < 2e-16 ***
## p2004          0.141971   0.007813  18.170  < 2e-16 ***
## yob           -0.020804   0.003658  -5.687 1.31e-08 ***
## city           0.044970   0.003585  12.545  < 2e-16 ***
## employ_20to64 -0.012767   0.003589  -3.557 0.000376 ***
## g2002.1        0.021979   0.025229   0.871 0.383662    
## p2004.1        0.025973   0.018914   1.373 0.169707    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.4477 on 16489 degrees of freedom
## Multiple R-squared:  0.06971,	Adjusted R-squared:  0.06914 
## F-statistic: 123.6 on 10 and 16489 DF,  p-value: < 2.2e-16
```
That is, now, after selecting by LASSO our most relevant variables, we note heterogeneity in some of our variables. However, we have a set of relevant variables different from the treatment that are also statistically significant. 

What is remaining is our estimation of CATE for post-selection LASSO. We can code it as the following, using the test sample:


```r
# Prediction on control group (X = 0)
postlasso_pred_control <- predict(sl_post_lasso, data.frame(X = zeros(nrow(W_test)), W_test, W_test*zeros(nrow(W_train)))[,c(kept_variables,kept_variables2)], onlySL = T)

# Prediction on control group (X = 1)
postlasso_pred_treated <- predict(sl_post_lasso, data.frame(X = ones(nrow(W_test)), W_test, W_test*ones(nrow(W_test)))[,c(kept_variables,kept_variables2)], onlySL = T)

# Estimating CATE with post-selection LASSO
cate_postlasso <- postlasso_pred_treated$pred - postlasso_pred_control$pred

# Plot cate_postlasso
plot(cate_postlasso)
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
# ATE
mean(cate_postlasso)
```

```
## [1] 0.08990288
```

```r
# Plot ATE
plot(mean(cate_postlasso))
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-7-2.png)<!-- -->
\section{3.3. Causal Trees}

Now we are going to predict the CATE from tree-based algorithms. There are a few packages that do this, but I like Susan Athey's `causalTree` and `grf`. The first is a general algorithm that builds a regression model and returns an \textit{rpart} object, implementing ideas from the CART (Classification and Regression Trees), by Breiman et al. The second implements the generalized random forest algorithm for causal forest, which uses a splitting tree-based rule to divide covariates based in the heterogeneity of treatment effects and, moreover, assumes honesty as one of main assumptions. 

In summary, we want to model

$$ Y_i = W_i + \theta X_i + \varepsilon $$
Where $c$ are possible covariates that brings hetrogeneity on the treatment $X_i$ for the outcome $Y_i$. The algorithm finds

$$ \hat{\tau}(W) = argmin_\tau \alpha_i(W) W_i(Y_i - \tau X_i)^2 $$
Where the weights $\alpha_i$ are estimated by a random forest algorithm. Let's begin by building our causal tree:


```r
# Witting the regression formula (to facilitate later)
tree_fml <- as.formula(paste("Y", paste(names(W_train), collapse = ' + '), sep = " ~ "))

# Building causal tree
causal_tree <- causalTree(formula = tree_fml,
                          data = data.frame(Y = Y_train,W_train),
                          treatment = X_train,
                          split.Rule = "CT",           # Causal Tree
                          split.Honest = FALSE,        # So far, we are not assuming honesty
                          split.alpha = 1,
                          cv.option = "CT",
                          cv.Honest = FALSE,
                          split.Bucket = TRUE,
                          bucketNum = 5,
                          bucketMax = 100,
                          minsize = 250                # Number of obs in treatment and control on leaf
                          )
```

```
## [1] 6
## [1] "CTD"
```

```r
rpart.plot(causal_tree, roundint = FALSE)
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-8-1.png)<!-- -->
\subsubsection{3.3.1. Honest Causal Trees}

Honest trees can be obtained when we use part of the sample (train) for building the leafs and part (test) to calculate the heterogeneity of treatment effects. The function `honest.causalTree` does the job:


```r
honest_tree <- honest.causalTree(formula = tree_fml,
                                 data = data.frame(Y=Y_train, W_train),
                                 treatment = X_train,
                                 est_data = data.frame(Y=Y_test, W_test),
                                 est_treatment = X_test,
                                 split.alpha = 0.5,
                                 split.Rule = "CT",
                                 split.Honest = TRUE,
                                 cv.alpha = 0.5,
                                 cv.option = "CT",
                                 cv.Honest = TRUE,
                                 split.Bucket = TRUE,
                                 bucketNum = 5,
                                 bucketMax = 100, # maximum number of buckets
                                 minsize = 250) # number of observations in treatment and control on leaf
```

```
## [1] 6
## [1] "CTD"
```

```r
rpart.plot(honest_tree, roundint = F)
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

Then, we prune the tree with cross validation, choosing the simplest tree that minimizes the objective function in a left-out sample:


```r
opcpid <- which.min(honest_tree$cp[, 4]) 
opcp <- honest_tree$cp[opcpid, 1]
honest_tree_prune <- prune(honest_tree, cp = opcp)

rpart.plot(honest_tree_prune, roundint = F)
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

That is, we have spitted our sample in order to maximize the heterogeneity in each node. That is the way these algorithms work: By splitting in sub-samples by heterogeneity, we are able to estimate with more accuracy the treatment effect. 

To estimate the standard errors on the leaves, we can use an OLS. The linear regression is specified such that the coefficients on the leaves are the treatment effects. 


```r
# Constructing factors variables for the leaves
leaf_test <- as.factor(round(predict(honest_tree_prune,
                                     newdata = data.frame(Y = Y_test, W_test),
                                     type = "vector"), 4))

# Run an OLS that estimate the treatment effect magnitudes and standard errors
honest_ols_test <- lm(Y ~ leaf + X * leaf - X -1, data = data.frame(Y = Y_test, X = X_test, leaf = leaf_test, W_test))

summary(honest_ols_test)
```

```
## 
## Call:
## lm(formula = Y ~ leaf + X * leaf - X - 1, data = data.frame(Y = Y_test, 
##     X = X_test, leaf = leaf_test, W_test))
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -0.3810 -0.3003 -0.3003  0.6710  0.6997 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## leaf0.051    0.329028   0.012100  27.191  < 2e-16 ***
## leaf0.0808   0.300252   0.004187  71.702  < 2e-16 ***
## leaf0.051:X  0.050972   0.029400   1.734    0.083 .  
## leaf0.0808:X 0.080778   0.010276   7.861 4.04e-15 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.4641 on 16496 degrees of freedom
## Multiple R-squared:  0.3191,	Adjusted R-squared:  0.3189 
## F-statistic:  1933 on 4 and 16496 DF,  p-value: < 2.2e-16
```

That is, all the heterogeneities in our pruned tree are relevant in our honest tree. 

But we still need to predict our CATE from our tree. This is easily done by the function predict:


```r
# Estimate CATE
cate_honesttree <- predict(honest_tree_prune, newdata = data.frame(Y = Y_test, W_test), type = "vector")

# Plot CATE
plot(cate_honesttree)
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

```r
# ATE
mean(cate_honesttree)
```

```
## [1] 0.07757872
```

```r
# Plot ATE
plot(mean(cate_honesttree))
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-12-2.png)<!-- -->

\subsection{3.4. Causal Forests}

Finally, we estimate the CATE with the causal forest algorithm. The method is similar to the R-learner, but with a splitting procedure using a tree-based algorithm. It uses a residual-residual approach do estimate the propensity score in order to find the conditional average treatment effect. For more information, check Jacob (2021).

Let's do in two ways. The first one will be using the `grf` package that estimates the CATE directly in the function. The second uses the `causalForest` package, estimating a random forest with honest causal trees. 

\subsection{3.4.1. Generalized Random Forest}

The `grf` algorithm assumes honesty as the main assumption. We can easily do this by fitting the causal forest with our training sample and predicting with our testing sample. Estimating our causal forest is simply:


```r
# Train our causal forest
cf_grf <- causal_forest(X = W_train,
                        Y = Y_train,
                        W = X_train)

# Get predictions from test sample
effects_cf_grf <- predict(cf_grf,W_test)

# Get effects
effects_cf_grf_pred <- predict(cf_grf,W_test)$predictions

# Histogram
hist(effects_cf_grf_pred)
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

```r
# Plot of treatment effects
plot(W_test[, 1], effects_cf_grf_pred, ylim = range(effects_cf_grf_pred, 0, 2), xlab = "x", ylab = "tau", type = "l")
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-13-2.png)<!-- -->

```r
# Estimate the CATE for the full sample
cate_grf <- average_treatment_effect(cf_grf, target.sample = "all")

cate_grf
```

```
##    estimate     std.err 
## 0.089492286 0.009709758
```

```r
# Estimate the CATE for the treated sample (CATT)
average_treatment_effect(cf_grf, target.sample = "treated")
```

```
##    estimate     std.err 
## 0.090869274 0.009679121
```

```r
# Best Linear Projection of the CATE
cate_best_grf <- best_linear_projection(cf_grf)

cate_best_grf
```

```
## 
## Best linear projection of the conditional average treatment effect.
## Confidence intervals are cluster- and heteroskedasticity-robust (HC3):
## 
##              Estimate Std. Error t value  Pr(>|t|)    
## (Intercept) 0.0894749  0.0098376  9.0952 < 2.2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

That is, `cate_best_grf` and `cate_grf` match each other, both when we predict the result using the `average_treatment_effect` function or when we use the `best_linear_projection`. Let's see if it holds without the `grf` package.

\subsubsection{3.4.2. causalTree package}

The same estimation can be obtained by the function `causalForest` inside the `causalTree` package:


```r
cf_causalTree <- causalForest(tree_fml,
                             data=data.frame(Y=Y_train, W_train), 
                             treatment=X_train, 
                             split.Rule="CT", 
                             split.Honest=T,  
                             split.Bucket=T, 
                             bucketNum = 5,
                             bucketMax = 100, 
                             cv.option="CT", 
                             cv.Honest=T, 
                             minsize = 2, 
                             split.alpha = 0.5, 
                             cv.alpha = 0.5,
                             sample.size.total = floor(nrow(Y_train) / 2), 
                             sample.size.train.frac = .5,
                             mtry = ceiling(ncol(W_train)/3), 
                             nodesize = 5, 
                             num.trees = 10, 
                             ncov_sample = ncol(W_train), 
                             ncolx = ncol(W_train))
```

```
## [1] "Building trees ..."
## [1] "Tree 1"
## [1] 6
## [1] "CTD"
## [1] "Tree 2"
## [1] 6
## [1] "CTD"
## [1] "Tree 3"
## [1] 6
## [1] "CTD"
## [1] "Tree 4"
## [1] 6
## [1] "CTD"
## [1] "Tree 5"
## [1] 6
## [1] "CTD"
## [1] "Tree 6"
## [1] 6
## [1] "CTD"
## [1] "Tree 7"
## [1] 6
## [1] "CTD"
## [1] "Tree 8"
## [1] 6
## [1] "CTD"
## [1] "Tree 9"
## [1] 6
## [1] "CTD"
## [1] "Tree 10"
## [1] 6
## [1] "CTD"
```

And, to estimate the CATE:


```r
cate_causalTree <- predict(cf_causalTree, newdata = data.frame(Y = Y_test, W_test), type = "vector")
```

```
## [1] 16500    10
```


\section{4. Comparing models}

Finally, we can compare all of our models (OLS with interaction terms, post-selection LASSO, Causal trees and Causal forest). Let's first see some histograms:


```r
# Creating a data frame withh all CATEs
het_effects <- data.frame(ols = cate_ols,
                          post_selec_lasso = cate_postlasso,
                          causal_tree = cate_honesttree,
                          causal_forest_grf = effects_cf_grf,
                          causal_forest_causalTree = cate_causalTree)

# Set range of x-axis
xrange <- range(c(het_effects[,1],het_effects[,2],het_effects[,3],het_effects[,4],het_effects[,5]))

# Set margins (two rows, five columns)
par(mfrow = c(1,5))

hist(het_effects[, 1], main = "OLS", xlim = xrange)
hist(het_effects[, 2], main = "Post-selection Lasso", xlim = xrange)
hist(het_effects[, 3], main = "Honest Causal tree", xlim = xrange)
hist(het_effects[, 4], main = "GRF Causal forest", xlim = xrange)
hist(het_effects[, 5], main = "causalTree Causal forest", xlim = xrange)
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

And to finalize, we can summary a table of results with each of CATEs:


```r
summary_stats <- do.call(data.frame, 
                         list(mean = apply(het_effects, 2, mean),
                              sd = apply(het_effects, 2, sd),
                              median = apply(het_effects, 2, median),
                              min = apply(het_effects, 2, min),
                              max = apply(het_effects, 2, max)))

summary_stats
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["mean"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["sd"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["median"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["min"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["max"],"name":[5],"type":["dbl"],"align":["right"]}],"data":[{"1":"0.09121995","2":"0.043849739","3":"0.09065545","4":"-0.06924895","5":"0.32668649","_rn_":"ols"},{"1":"0.08990288","2":"0.015283956","3":"0.08263584","4":"0.06065641","5":"0.10860866","_rn_":"post_selec_lasso"},{"1":"0.07757872","2":"0.009226247","3":"0.08077787","4":"0.05097213","5":"0.08077787","_rn_":"causal_tree"},{"1":"0.09069419","2":"0.030955390","3":"0.09068478","4":"-0.03018220","5":"0.24469023","_rn_":"predictions"},{"1":"0.09534748","2":"0.074583769","3":"0.09490361","4":"-0.16502486","5":"0.37749714","_rn_":"causal_forest_causalTree"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>
From the histograms and the summary statistics, it seems that the causal forest from the `grf` package yields most heterogeneity, once we have higher standard deviation among our treatment effects.

We can also compare the methods by looking at the minimum square error (MSE) on a test set using the transformed outcome (which I call $Y^*$) as a proxy for the true treatment effect. For this, we need to construct the propensity score ($E(X = 1)$) of our treatment in our test sample. Let's code it:


```r
# Construct propensity score from randomized experiment
prop_score <- mean(X_test)

# Construct Y_star in our test sample
Y_star <- X_test * (Y_test / prop_score) - (1 - X_test) * (Y_test / (1 - prop_score))

## MSEs

# OLS with interaction
MSE_ols <- mean((Y_star - cate_ols)^2)

# Post-selection LASSO
MSE_lasso <- mean((Y_star - cate_postlasso)^2)

# Honest Tree
MSE_causalTree <- mean((Y_star - cate_honesttree)^2)

# Causal Forest GRF
MSE_cf_grf <- mean((Y_star - cate_grf)^2)

# Causal Forest causalTree
MSE_cf_causalTree <- mean((Y_star - cate_causalTree)^2)

# Create data frame with all MSEs
performance_MSE <- data.frame(matrix(rep(NA,1), nrow = 1, ncol = 1))
rownames(performance_MSE) <- c("OLS")
colnames(performance_MSE) <- c("MSE")

# Load in results
performance_MSE["OLS","MSE"] <- MSE_ols
performance_MSE["Post-selection LASSO","MSE"] <- MSE_lasso
performance_MSE["Honest Tree","MSE"] <- MSE_causalTree
performance_MSE["Causal Forest GRF","MSE"] <- MSE_cf_grf
performance_MSE["Causal Forest causalTree","MSE"] <- MSE_cf_causalTree

# Setting range
xrange2 <- range(performance_MSE$MSE - 2*sd(performance_MSE$MSE), 
                 performance_MSE$MSE,
                 performance_MSE$MSE + 2*sd(performance_MSE$MSE))

# Create plot
MSEplot <- ggplot(performance_MSE) + 
  geom_bar(mapping = aes(x = factor(rownames(performance_MSE), 
                                    levels = rownames(performance_MSE)), 
                         y = MSE),
           stat = "identity", fill = "gray44", width=0.7, 
           position = position_dodge(width=0.2)) + 
  theme_bw() + 
  coord_cartesian(ylim=c(xrange2[1], xrange2[2])) +
  theme(axis.ticks.x = element_blank(), axis.title.x = element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        plot.background = element_blank(),
        axis.text.x = element_text(angle = -90, hjust = 0, vjust = 0.5)) +
  ylab("MSE out-of-sample") + 
  ggtitle("Comparing performance based on MSE") +
  theme(plot.title = element_text(hjust = 0.5, face ="bold", 
                                  colour = "black", size = 14))

# Plot
MSEplot
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-18-1.png)<!-- -->

From the MSE analysis, it seems that both causal forest models perform worse than simpler models like OLS, post-selection LASS) and honest tree. One explanation could be that since we have little heterogeneity in our model, simpler models do best. 

\section{6. Interpretable Machine Learning}

Finally, let's do a visual analysis on the feature importance of our variables in the voters' turnout. For this, we will use the `iml`, a nice package that for interpretable machine learning in R. 

One of the most interesting features from the `iml` package is the possibility of plotting Shapley Values - a method from coalitional game theory - tells us how to fairly distribute the "payout" among the covariates. 

\subsection{6.1. Feature importance}

First, let's create a `Predictor` object. This is necessary since the `iml` package uses R6 classes, where New objects must be created from estimated machine learning models that holds the model itself and the data. This is done as the following:


```r
# Using the iml Predictor() container
cf_predictor <- Predictor$new(cf_grf, data = W_train, y = Y_train)
```


We can measure how each feature is important for the prediction of our causal forest with the function `FeatureImp`. The feature importance measure works by shuffling each feature and measuring how much the performance drops. 


```r
cf_importance <- FeatureImp$new(cf_predictor, loss = "mse")

plot(cf_importance)
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-20-1.png)<!-- -->

```r
cf_importance$results
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":["feature"],"name":[1],"type":["chr"],"align":["left"]},{"label":["importance.05"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["importance"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["importance.95"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["permutation.error"],"name":[5],"type":["dbl"],"align":["right"]}],"data":[{"1":"p2004","2":"1.0010331","3":"1.0011156","4":"1.0013114","5":"0.2654978"},{"1":"yob","2":"1.0003777","3":"1.0005522","4":"1.0011477","5":"0.2653484"},{"1":"g2002","2":"1.0002292","3":"1.0002469","4":"1.0002902","5":"0.2652674"},{"1":"percent_white","2":"0.9999998","3":"1.0001587","4":"1.0003595","5":"0.2652440"},{"1":"hh_size","2":"0.9997920","3":"1.0000724","4":"1.0002546","5":"0.2652211"},{"1":"sex","2":"1.0000000","3":"1.0000590","4":"1.0002659","5":"0.2652176"},{"1":"g2000","2":"1.0000015","3":"1.0000073","4":"1.0000123","5":"0.2652039"},{"1":"percent_asian","2":"0.9997747","3":"0.9998901","4":"0.9999719","5":"0.2651728"},{"1":"p2002","2":"0.9997648","3":"0.9998148","4":"0.9999011","5":"0.2651528"},{"1":"p2000","2":"0.9997296","3":"0.9997808","4":"0.9997954","5":"0.2651438"},{"1":"median_age","2":"0.9995145","3":"0.9997371","4":"0.9998807","5":"0.2651322"},{"1":"percent_hispanicorlatino","2":"0.9993169","3":"0.9994718","4":"0.9996236","5":"0.2650619"},{"1":"percent_black","2":"0.9989741","3":"0.9990174","4":"0.9993285","5":"0.2649414"},{"1":"city","2":"0.9985865","3":"0.9988981","4":"0.9990514","5":"0.2649097"},{"1":"highschool","2":"0.9980634","3":"0.9982575","4":"0.9983134","5":"0.2647398"},{"1":"percent_62yearsandover","2":"0.9981188","3":"0.9981638","4":"0.9982750","5":"0.2647150"},{"1":"percent_male","2":"0.9979536","3":"0.9980539","4":"0.9982518","5":"0.2646858"},{"1":"totalpopulation_estimate","2":"0.9977145","3":"0.9978170","4":"0.9981053","5":"0.2646230"},{"1":"employ_20to64","2":"0.9972243","3":"0.9975068","4":"0.9975970","5":"0.2645407"},{"1":"bach_orhigher","2":"0.9965360","3":"0.9966787","4":"0.9968007","5":"0.2643211"},{"1":"median_income","2":"0.9949955","3":"0.9950827","4":"0.9952707","5":"0.2638979"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

\subsection{6.2. Feature effects}

Is interesting to know also how the features influence the predicted outcome. This is done by the function `FeatureEffect`.

\subsection{6.3. Shapley Values}


```r
cf_shapley <- Shapley$new(cf_predictor, x.interest = W_train[1,])

cf_shapley$plot()
```

![](Tutorial-1---HTE-in-Randomized-Data_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

There are still a lot of features from the `iml` package. The next version from this PDF will deal with them.

Moreover, \textit{Tutorial 2 - HTE in Randomized Panel Data with Machine Learning Tecniques} brings the `grf` application in randomized panel data, when we account for individual fixed effects. You can find it at my website (https://sites.google.com/view/victor-hugo-alexandrino/) and GitHub (https://github.com/victoralexs)


\section{References}

https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html

https://gsbdbi.github.io/ml_tutorial/

https://ml-in-econ.appspot.com

https://github.com/QuantLet/Meta_learner-for-Causal-ML/blob/main/GRF/Causal-Forest.R

https://grf-labs.github.io/grf/

https://www.markhw.com/blog/causalforestintro

https://lost-stats.github.io/Machine_Learning/causal_forest.html






