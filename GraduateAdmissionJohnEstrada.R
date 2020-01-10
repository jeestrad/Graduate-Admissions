################################
# DOWNLOAD THE DATASET FROM KAGGLE
################################

################################
# CITATION OF THE DATASET
#
#Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions,
#IEEE International Conference on Computational Intelligence in Data Science 2019
#
################################


# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("rpart", repos = "http://cran.us.r-project.org")

# Lets load the libraries

library(data.table)
library(tidyverse)
library(caret)
library(knitr)
library(purrr)
library(rpart)

# Download the dataset directly from my repository. Originally, the dataset was downloaded
# from Kaggle but as I am not sure whether the grader has a Kaggle account, I decided to upload
# the dataset to make it available to anyone.
# The link for the Kaggle source is "https://www.kaggle.com/mohansacharya/graduate-admissions/version/2#Admission_Predict_Ver1.1.csv"



url <- "https://raw.githubusercontent.com/jeestrad/Graduate-Admissions/master/Admission_Predict_Ver1.1.csv"
admissions <- read_csv(url)

# Let's modifiy the column names to be able to work with them in a more efficient way
colnames(admissions) <- c("Serial", "GRE", "TOEFL", "Urating", "SOP", "LOR", "CGPA", "Research", "Chance of Admit")

################################
# MEANING OF THE COLUMN NAMES
################################

# Serial No: A consecutive number assigned to the candidate on the poll
# GRE Score: GRE Scores ( out of 340 )
# TOEFL Score: TOEFL Scores ( out of 120 )
# Univerisity Rating: University Rating ( out of 5 ), being 5 the Top Univeristy and 1 the ones at the bottom.
# SOP: Statement of Purpose Strength ( out of 5 )
# LOR: Letter of Recommendation Strength ( out of 5 )
# CGPA: Undergraduate GPA ( out of 10 )
# Research: Research Experience ( either 0 or 1 )
# Chance of Admit: This was answered by the candidate as his/her chances to be admitted 

head(admissions)

# The Chance of Admit is the chance that each applicant thought of being accepted to
# the Univerisity applied. Unfortunately, the actual admitted or not is not part of the dataset
# but the information by the dataset is great, useful and can provide a formidable insight. 
# Because of that, I can apply a cutoff equal to the 
# average of Chance of Admit, arbitrary. In that sense, simulating as that student above the cutoff was admitted
# and those below the cuttoff do not. 
# After that, I deleted the column used the column "Chance of Admit" to avoid making it a predictor
# on the models. Also, I made admitted equal to 1 and not admitted equal to 0.

#Let's find that cutoff
mean(admissions$`Chance of Admit`)

# Now, apply a logical to produce admitted and not admitted, 1 and 0, respectively.
admitted <- ifelse(admissions$`Chance of Admit` >= 0.72, 1, 0)
admissions <- mutate(admissions, admitted)

# We do not want the column used for producing the cutoff (Chande of Admit) to be one of our predictors
# so we remove it
admissions <- admissions[-9]


# A first glimpse of the header seems like columns "Research" and "Admitted" are identical,
# let's take a look if that if that is the case.
identical(admissions$Research,admissions$admitted)

################################
# CREATE A PARTITION FOR VALIDATION SET AS 10% OF THE DATA
################################
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
#if using R 3.5 or earlier, use `set.seed(1)` instead
#set.seed(1)
test_index <- createDataPartition(y = admissions$admitted, times = 1, p = 0.1, list = FALSE)
admit <- admissions[-test_index,]
validation <- admissions[test_index,]
# Therefore, the admit data set will be the one used for training purposes and validation the dataset
# where the best model will be finally tested to determine the accuracy of the model.

# Let's find the proportion of admitted from the validation dataset
mean(validation$admitted)

################################
# CREATE A PARTITION FOR TRAIN_SET AND TEST_SET 10% OF THE DATA
################################
# We are going to create an additional partition where we can test different models. Once the best
# model is selected (based on accuracy), it will be tested on the validation dataset to determine the final
# accuracy.

set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
#if using R 3.5 or earlier, use `set.seed(1)` instead
#set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
#if using R 3.5 or earlier, use `set.seed(1)` instead
#set.seed(1)
test_index <- createDataPartition(y = admit$admitted, times = 1, p = 0.1, list = FALSE)
train_set <- admissions[-test_index,]
test_set <- admissions[test_index,]



################################
# DATA EXPLORATION AND DATA CLEANING
################################
# Let's find the proportion of admitted from the test dataset
mean(test_set$admitted)
mean(train_set$admitted)

dim_train <- dim(train_set)
dim_train
dim_test <- dim(test_set)
dim_test

# Lets look at the GRE score
mean_GRE_train <- mean(train_set$GRE)
mean_GRE_train
hist(train_set$GRE)

# Lets look at the TOEFL
mean_TOEFL_train <- mean(train_set$`TOEFL`)
mean_TOEFL_train
hist(train_set$`TOEFL`)

# Strength of SOP
hist(train_set$`SOP`)

# CGPA
hist(train_set$CGPA)

# Let see the GRE score depending on the rating of University
train_set %>% group_by(`Urating`) %>% ggplot(aes(`Urating`, `GRE`, group= `Urating`)) + geom_boxplot()
# We can see some outliers with low GRE score applying to Universities of Ranking 4 and 5.

# Let see the TOEFL score depending on the rating of University
train_set %>% group_by(`Urating`) %>% ggplot(aes(`Urating`, `TOEFL`, group= `Urating`)) + geom_boxplot()

# Let see the CGPA score depending on the rating of University
train_set %>% group_by(`Urating`) %>% ggplot(aes(`Urating`, CGPA, group= `Urating`)) + geom_boxplot()

# To this point, it seems that GRE, TOEFL, and CGPA, are strongly correlated to the admission rate. Let's 
# evaluate the correlations for each:

print("correlation_GRE_admit")
correlation_GRE_admit <- cor(train_set$`GRE`, train_set$admitted)
print("correlation_TOEFL_admit")
correlation_TOEFL_admit <- cor(train_set$`TOEFL`, train_set$admitted)
print("correlation_CGPA_admit")
correlation_CGPA_admit <- cor(train_set$CGPA, train_set$admitted)
print("correlation_SOP_admit")
correlation_SOP_admit <- cor(train_set$`SOP`, train_set$admitted)
print("correlation_LOR_admit")
correlation_LOR_admit <- cor(train_set$`LOR`, train_set$admitted)
print("correlation_Research_admit")
correlation_Research_admit <- cor(train_set$Research, train_set$admitted)

tibble(correlation_GRE_admit, correlation_TOEFL_admit, correlation_CGPA_admit, correlation_SOP_admit,
       correlation_LOR_admit,correlation_Research_admit)

# Most of predictors are highly correlated to the admission rate. It seems that both the GRE and CGPA have
# even a stronger correlation than the TOEFL. However, does it make any predictor stronger than the other? We
# wil find out this later. Let's keep exploring the data.

# How is the relationships between GRE and CGPA
train_set %>% group_by(`GRE`) %>% ggplot(aes(`GRE`, CGPA)) + geom_smooth()
cor(train_set$`GRE`, train_set$CGPA)

# How is the relationships between GRE and SOP
cor(train_set$`GRE`,train_set$SOP)
train_set %>% group_by(`GRE`) %>% ggplot(aes(`GRE`, SOP)) + geom_smooth()

# How is the relationships between GRE and LOR
cor(train_set$`GRE`,train_set$LOR)
train_set %>% group_by(`GRE`) %>% ggplot(aes(`GRE`, LOR)) + geom_smooth()

# To this point we can observe that some variables are highly correlated on to each other more than others.
# Also, the the previous chart suggest the the GRE and CGPA may be powerful predictors for admission

################################
# METHODS
################################

# After analyzing the data, we can start testing some of the models. The key parameter for determining
# which model will be selected for testing on the validation dataset will be the one with highest accuracy
# all the models will be evaluated on the test_set to have a point of comparison among them. 

# NAIVE MODEL
################################
# NAIVE MODEL. Test our luck here by estimating the admitted by randomly sampling
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
y_hat <- sample(c(0, 1), length(test_index), replace = TRUE)
accuracy_naive_model <- mean(y_hat == test_set)
print("accuracy_naive_model")
accuracy_naive_model
# Pretty low. Almost half and half chances of picking up the right answer. Lets try other models then.

# GRE MODEL
################################
# GRE MODEL. From the previous data exploration GRE seem to be highly correlated to the admission rate
# let's produce an estimation using only that predictor. However, GRE is not categorical, so which cutoff
# can we use here? Let's use as cutoff the average GRE minus two standard deviations of the average obtained
#from those who were admitted. This parameter
# has been selected by me, arbitrary. 
a <- test_set %>% filter(admitted == 1)
# Table a only includes the values of those admitted.
cutoff_GRE_admit <- mean(a$GRE)-2*sd(a$GRE)
print("cutoff_GRE_admit")
cutoff_GRE_admit

GRE_model <- ifelse(test_set$GRE >= cutoff_GRE_admit, 1, 0) 
GRE_model_accuracy <- mean(GRE_model == test_set$admitted)
print("GRE_model_accuracy")
GRE_model_accuracy
# Not bad, even better than (Naive)

# CPGA MODEL
################################
# CGPA also presented a high correlation to the admission rate. The cutoff again will 
# be the average CGPA minus two standard deviations, arbitrary, for those admitted. I anticipate an accuracy fairly good, too.

cutoff_CGPA_admit <- mean(a$CGPA)- 2*sd(a$CGPA)
cutoff_CGPA_admit

CGPA_model <- ifelse(test_set$CGPA >= cutoff_CGPA_admit, 1, 0) 
accuracy_CGPA_model <- mean(CGPA_model == test_set$admitted)
accuracy_CGPA_model
# This is a slightly good improvement here even in comparison to the GRE

# TOEFL MODEL
################################
# TOEFL had a less strong correlation. Let's see if the accuracy is lower, effectively

cutoff_TOEFL_admit <- mean(a$TOEFL) - 2*sd(a$TOEFL)
cutoff_TOEFL_admit

# Let's try using directly as cutoff the average CGPA score of those admitted
TOEFL_model <- ifelse(test_set$TOEFL >= cutoff_TOEFL_admit, 1, 0) 
accuracy_TOEFL_model <- mean(TOEFL_model == test_set$admitted)
accuracy_TOEFL_model
# Surprisingly, despite of the lowwe correlation, its accuracy seems even better than GRE and CGPA.

# TOEFL_CGPA_GRE MODEL
################################
# Let's try a model that includes the predictors with highest correlation together (TOEFL, GRE, CGPA), considering again
# as the cutoff the mean of each predictor minus two times the standard deviation, for those admitted.

TOEFL_CGPA_GRE_mix_model <- ifelse(test_set$TOEFL >= cutoff_TOEFL_admit &
                                     test_set$CGPA >= cutoff_CGPA_admit &
                                     test_set$GRE >= cutoff_GRE_admit
                                     , 1, 0) 
accuracy_TOEFL_CGPA_GRE_mix_model <- mean(TOEFL_CGPA_GRE_mix_model == test_set$admitted)
accuracy_TOEFL_CGPA_GRE_mix_model



# SUMMARY OF ACCURACIES OF MODELS SO FAR
################################

summary <- tibble(model = "naive_model", accuracy = accuracy_naive_model)
summary <- bind_rows(summary, tibble(model = "GRE_model", accuracy = GRE_model_accuracy))
summary <- bind_rows(summary, tibble(model = "CGPA_model", accuracy = accuracy_CGPA_model))
summary <- bind_rows(summary, tibble(model = "TOEFL_model", accuracy = accuracy_TOEFL_model)) 
summary <- bind_rows(summary, tibble(model = "TOEFL_CGPA_GRE_mix_model", accuracy = accuracy_TOEFL_CGPA_GRE_mix_model))
summary

# CONFUSION MATRIX
print("Confusion Matrix GRE_model")
confusionMatrix(data = factor(GRE_model), reference = factor(test_set$admitted))
print("Confusion Matrix CGPA_model")
confusionMatrix(data = factor(CGPA_model), reference = factor(test_set$admitted))
print("Confusion Matrix TOEFL_model")
confusionMatrix(data = factor(TOEFL_model), reference = factor(test_set$admitted))
print("Confusion Matrix TOEFL_CGPA_GRE_mix_model")
confusionMatrix(data = factor(TOEFL_CGPA_GRE_mix_model), reference = factor(test_set$admitted))

# The highest balanced accuracy (0.889) and sensitivity (0.854) were obtained by mix_model, highest specificity
# is however higher on the other modules.

# F1_SCORES
print("F1 Score GRE_model")
F_meas(data = factor(GRE_model), reference = factor(test_set$admitted))
print("F1 Score CGPA_model")
F_meas(data = factor(CGPA_model), reference = factor(test_set$admitted))
print("F1 Score TOEFL_model")
F_meas(data = factor(TOEFL_model), reference = factor(test_set$admitted))
print("F1 Score TOEFL_CGPA_GRE_mix_model")
F_meas(data = factor(TOEFL_CGPA_GRE_mix_model), reference = factor(test_set$admitted))

# Again, the F1 score is maximum for the mix model

# The mix_model performed very good, let's see if we can use other models that can increase the accuracy.


# QUADRATIC DISCRIMINANT ANALYSIS
################################
#We will use the caret package from this point on for simplicity and homogeneity in the code.

# Let's try the TOEFL Score as the unique predictor
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_QDA_TOEFL <- train(factor(admitted) ~ TOEFL, method = "qda", data = train_set)
Admitted_QDA_TOEFL <- predict(fit_QDA_TOEFL, test_set)
accuracy_QDA_TOEFL <- mean(Admitted_QDA_TOEFL == factor(test_set$admitted))
accuracy_QDA_TOEFL
print("Confusion Matrix TOEFL_CGPA_GRE_mix_model")
confusionMatrix(data = (Admitted_QDA_TOEFL), reference = factor(test_set$admitted))

# The accuracy seems to be pretty high. Let's inspect quickly how it performs with the validation dataset
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_QDA_TOEFL <- train(factor(admitted) ~ TOEFL, method = "qda", data = admit)
Admitted_QDA_TOEFL <- predict(fit_QDA_TOEFL, validation)
mean(Admitted_QDA_TOEFL == factor(validation$admitted))
# The accuracy dropped considerably, suggesting overfitting of the model. Let's keep evaluation other models

# Let's use QDA with the predictors that presented higher correlation (TOEFL, GRE, CGPA)
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_QDA_mix <- train(factor(admitted) ~ GRE + CGPA + TOEFL, method = "qda", data = train_set)
Admitted_QDA_mix <- predict(fit_QDA_mix, test_set)
accuracy_QDA_mix <- mean(Admitted_QDA_mix == factor(test_set$admitted))
accuracy_QDA_mix

# Let's use QDA with ALL the predictors present
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_QDA_all <- train(factor(admitted) ~ GRE + TOEFL + CGPA + Research + SOP + LOR + Urating, method = "qda", data = train_set)
Admitted_QDA_all <- predict(fit_QDA_all, test_set)
accuracy_QDA_all <- mean(Admitted_QDA_all == factor(test_set$admitted))
accuracy_QDA_all

# LINEAR DISCRIMINANT ANALYSIS
################################
# Let's use LDA with the predictors that presented higher correlation (TOEFL, GRE, CGPA)
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_LDA_all <- train(factor(admitted) ~ GRE + CGPA + TOEFL, method = "lda", data = train_set)
Admitted_LDA_all <- predict(fit_LDA_all, test_set)
accuracy_LDA_all <- mean(Admitted_LDA_all == factor(test_set$admitted))
accuracy_LDA_all

# TO THIS POINT EITHER LDA AND QDA BOTH HAVE AN ACCURACY AROUND 0.889

# GENERALIZED LINEAR MODEL
################################

# Let's use GLM with the highest correlated predictors
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_GLM_mix <- train(factor(admitted) ~ GRE + TOEFL + CGPA, method = "glm", data = train_set)
Admitted_GLM_mix <- predict(fit_GLM_mix, test_set)
accuracy_GLM_mix <- mean(Admitted_GLM_mix == factor(test_set$admitted))
accuracy_GLM_mix

#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_GLM_all <- train(factor(admitted) ~ GRE + TOEFL + CGPA + Research + SOP + LOR + Urating, method = "glm", data = train_set)
Admitted_GLM_all <- predict(fit_GLM_all, test_set)
accuracy_GLM_all <- mean(Admitted_GLM_all == factor(test_set$admitted))
accuracy_GLM_all
# Again here, a good accuracy is displayed. Let's quickly evaluate on the validation dataset

set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_GLM_all <- train(factor(admitted) ~ GRE + TOEFL + CGPA + Research + SOP + LOR + Urating, method = "glm", data = admit)
Admitted_GLM_all <- predict(fit_GLM_all, validation)
accuracy_GLM_all <- mean(Admitted_GLM_all == factor(validation$admitted))
accuracy_GLM_all

# kNN Model
################################
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_knn_all <- train(factor(admitted) ~ GRE + TOEFL + CGPA + Research + SOP + LOR + Urating, method = "knn", data = train_set, tuneGrid = data.frame(k = seq(3, 20, 1)))
accuracy_knn_all <- confusionMatrix(predict(fit_knn_all, test_set), as_factor(test_set$admitted))$overall["Accuracy"]
accuracy_knn_all
plot(fit_knn_all)
best_k <- fit_knn_all$bestTune
best_k
# Here it seems that we obtained a good accuracy but not notably better than the provious models.

# kNN and Cross-Validation
################################
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
control <- trainControl(method = "cv", number = 5, p = .1)
fit_knn_cv <- train(factor(admitted) ~ GRE + TOEFL + CGPA + Research + SOP + LOR + Urating, method = "knn", 
                    data = train_set,
                    tuneGrid = data.frame(k = seq(3, 20, 1)),
                    trControl = control)
accuracy_knn_cv <- confusionMatrix(predict(fit_knn_cv, test_set), as_factor(test_set$admitted))$overall["Accuracy"]
accuracy_knn_cv
plot(fit_knn_cv)
best_k <- fit_knn_cv$bestTune
best_k
# the accuracy here seems to have improved.

# TREE CLASSIFICATION MODEL
################################
# Perhaps this tree may provide a insight for potential graduate students to determine their possibilities
# of getting accepted to a particular university. Let's see

#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_rpart <- train(factor(admitted) ~ GRE + TOEFL + CGPA + Research + SOP + LOR + Urating, method = "rpart", data = train_set, tuneGrid = data.frame(cp = 0.001))
best_cp <- fit_rpart$bestTune
best_cp
accuracy_rpart <- confusionMatrix(predict(fit_rpart, test_set), as_factor(test_set$admitted))$overall["Accuracy"]
accuracy_rpart
plot(fit_rpart$finalModel, margin = 0.01)
text(fit_rpart$finalModel, cex = 0.9)
# Although the accuracy of the model is not higher than kNN, the tree provides an insightful data
# for those applicants willing to apply and get accepted. Unfortunatelly, the tree does not show the
# rating of the university. One workaround here (not included) would be grouping by Urating and obtaining
# trees for each one of them.


# RANDOM FOREST
################################

# Select the best parameter of mtry
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_rf <- train(as_factor(admitted) ~ ., method = "rf", data = train_set, tuneGrid = data.frame(mtry = seq(1, 8, 0.1)), ntree = 50)
confusionMatrix(predict(fit_rf, test_set), as_factor(test_set$admitted))$overall["Accuracy"]
plot(fit_rf)
best_mtry <- fit_rf$bestTune
best_mtry


# Select the appropiate number of trees
ntree <- seq(1, 800, 50)
fit_rf <- sapply(ntree, function(n){
  train(as_factor(admitted) ~ ., method = "rf", data = train_set, tuneGrid = data.frame(mtry = best_mtry), ntree = n)$results$Accuracy
})
qplot(ntree,fit_rf)
best_tree <- ntree[which.max(fit_rf)]
best_tree

# Then, the final model will be with the best MTRY and best NTREE
fit_rf <- train(as_factor(admitted) ~ ., method = "rf", data = train_set, tuneGrid = data.frame(mtry = best_mtry), ntree =best_tree)
accuracy_rf <- confusionMatrix(predict(fit_rf, test_set), as_factor(test_set$admitted))$overall["Accuracy"]
accuracy_rf
varImp(fit_rf)

# RESULTS - SUMMARY OF ACCURACIES ON TRAINED MODELS
################################
summary <- tibble(model = "naive_model", accuracy = accuracy_naive_model)
summary <- bind_rows(summary, tibble(model = "GRE_model", accuracy = GRE_model_accuracy))
summary <- bind_rows(summary, tibble(model = "CGPA_model", accuracy = accuracy_CGPA_model))
summary <- bind_rows(summary, tibble(model = "TOEFL_model", accuracy = accuracy_TOEFL_model)) 
summary <- bind_rows(summary, tibble(model = "TOEFL_CGPA_GRE_mix_model", accuracy = accuracy_TOEFL_CGPA_GRE_mix_model)) 
summary <- bind_rows(summary, tibble(model = "QDA_TOEFL", accuracy = accuracy_QDA_TOEFL)) 
summary <- bind_rows(summary, tibble(model = "QDA_mix", accuracy = accuracy_QDA_mix)) 
summary <- bind_rows(summary, tibble(model = "QDA_all", accuracy = accuracy_QDA_all)) 
summary <- bind_rows(summary, tibble(model = "LDA_all", accuracy = accuracy_LDA_all)) 
summary <- bind_rows(summary, tibble(model = "GLM_mix", accuracy = accuracy_GLM_mix)) 
summary <- bind_rows(summary, tibble(model = "GLM_all", accuracy = accuracy_GLM_all)) 
summary <- bind_rows(summary, tibble(model = "kNN_all", accuracy = accuracy_knn_all)) 
summary <- bind_rows(summary, tibble(model = "kNN_CV", accuracy = accuracy_knn_cv)) 
summary <- bind_rows(summary, tibble(model = "REGRESSION TREES", accuracy = accuracy_rpart)) 
summary <- bind_rows(summary, tibble(model = "RANDOM FOREST", accuracy = accuracy_rf)) 
summary
print("***being all: using all predictors, being mix: using only GRE+CGPA+TOEFL")    

#################
#ANALYSIS
#################

# QDA_TOEFL ON VALIDATION
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_QDA_TOEFL <- train(factor(admitted) ~ TOEFL, method = "qda", data = admit)
Admitted_QDA_TOEFL <- predict(fit_QDA_TOEFL, validation)
accuracy_QDA_TOEFL_validation <- mean(Admitted_QDA_TOEFL == factor(validation$admitted))
accuracy_QDA_TOEFL_validation
#The accuracy dropped notably.

# KNN_CV ON VALIDATION
#set.seed(1)
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
control <- trainControl(method = "cv", number = 5, p = .1)
fit_knn_cv <- train(factor(admitted) ~ GRE + TOEFL + CGPA + Research + SOP + LOR + Urating, method = "knn", 
                    data = admit,
                    tuneGrid = data.frame(k = best_k),
                    trControl = control)
accuracy_kNN_CV_validation <- confusionMatrix(predict(fit_knn_cv, validation), as_factor(validation$admitted))$overall["Accuracy"]
accuracy_kNN_CV_validation
# The accuracy decreased slightly, probably due to some minor overfitting.

# RANDOM FOREST ON VALIDATION
set.seed(1, sample.kind="Rounding") # if using a later version than R 3.5
fit_rf_final <- train(as_factor(admitted) ~ ., method = "rf", data = admit, tuneGrid = data.frame(mtry = best_mtry), ntree = best_tree)
accuracy_RF_validation <- confusionMatrix(predict(fit_rf_final, validation), as_factor(validation$admitted))$overall["Accuracy"]
accuracy_RF_validation
varImp(fit_rf_final)

summary <- tibble(model = "QDA_TOEFL", accuracy = accuracy_QDA_TOEFL_validation)
summary <- bind_rows(summary, tibble(model = "kNN_CV", accuracy = accuracy_kNN_CV_validation))
summary <- bind_rows(summary, tibble(model = "RANDOM FOREST", accuracy = accuracy_RF_validation))
summary

# ANALYSIS 
# The accuracy is substancially lower with the validation set than that obtained with the train and test sets in the three methods.
# The validation accuracy is lower because perhaps I've made it artificially harder for the train_set to give the right asnwer that it
# ended up overfitting the training data.

# Another possible reason might be that perhaps the validation set was not very representative in regard to the training set. In such case,
# a larger dataset would be useful to tune the models even more.

# Finally, based on the overall accuracy obtained here, the best model is the cross-validated K-nearest neighbors with an accuracy on the validation 
# of 0.88.




########
# NOTE TO THE GRADER: IF YOU RUN THE CODE AND 18 WARNING APPEAR, THEY CORRESPOND TO SETTING THE SEEDS()
#######

