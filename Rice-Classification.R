## RICE CLASSIFICATION USING LOGISTIC REGRESSION, RANDOM FOREST, CLASSIFICATION TREE, ADA BOOSTING, KNN, XGB and GBM
# load the necessary libraries
library(mice)
library(corrgram)
library(caTools)
library(pROC)
library(class)
library(caret)
library(corrplot)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(vip)
library(adabag)
library(ada)
library(xgboost)
library(gbm)

# load the data
df <- read.csv("rice_class.arff", header=FALSE, comment.char = "@", skip = 4)

# name the cols
colnames(df) <- c("Area", # in px (unit)
                  "Perimeter", # in px (unit)
                  "Major_Axis_Length",
                  "Minor_Axis_Length",
                  "Eccentricity",
                  "Convex_Area",
                  "Extent",
                  "Class")

# check the str of the df
str(df)

# change the class of the response col
df$Class <- as.factor(df$Class)

# check the missing values
md.pattern(df) # completely observed

# check the class col
table(df$Class) # ~ statistically balanced

# correlation graph
corr.matrix <- cor(df [, -8])
corrplot(corr.matrix, method = "number")

# drop the highly correlated cols (perimeter, Convex_Area, Major_Axis_Length) to prevent multicollinearity
df <- df [, -2]
df <- df [, -5]
df <- df [, -2]

# check again the correlation graph
corr.matrix <- cor(df [, -5])
corrplot(corr.matrix, method = "number") # looks good!

# test & train split
split <- sample.split(df$Class, SplitRatio = 0.7)
test <- subset(df, split == FALSE)
train <- subset(df, split == TRUE)

# dim of the train set
dim(train)

# dim of the test set
dim(test)

# multiple plots of the observations
pairs(~ Area + Minor_Axis_Length + Eccentricity + Extent, data = df)

## LOGISTIC REGRESSION
# building logistic regression model
glm.model <- glm(Class ~ ., data = train, family = binomial(logit))

# summary of the model (glm.model)
summary(glm.model) # cols with (*) are statistically significant

# perform stepwise variable selection to prevent possible overfitting
new.step.model <- step(glm.model)

# predictions of the new step model
lgm.preds.nsm <- predict(new.step.model, newdata = test, type = "response")

# only 0s and 1s of the new step model
lgm.preds.fitted.nsm <- ifelse(lgm.preds.nsm > 0.5, "Osmancik", "Cammeo")

# convert the lgm.preds.fitted into factor to be able to use in the confusion matrix
lgm.preds.fitted.nsm <- as.factor(lgm.preds.fitted.nsm)

# confusion matrix of the new step model model (new.step.model)
confusionMatrix(lgm.preds.fitted.nsm, reference = test$Class, positive = "Osmancik" )

# ROC
roc.data <- roc(test$Class, lgm.preds.nsm)
roc.data

# ROC plot
plot(roc.data)

## RANDOM FOREST
# creating parameter grid for random forest model
param.grid.rf <- expand.grid(mtry = c(1,2,3,4,5))

# setting cross - validation parameters with the control function
ctrl.rf <- trainControl(method = "cv", 
                        number = 5)

# select the best parameters for random forest
parameter.search.rf <- train(Class ~.,
                             data = train,
                             method = "rf",
                             trControl = ctrl.rf,
                             tuneGrid = param.grid.rf)

# best tune
parameter.search.rf$bestTune

# building a rf with the best tune parameters
rf.model <- randomForest(Class ~., train, mtry = parameter.search.rf$bestTune$mtry, ntree = 10)

# predictions of the RF model
rf.preds <- predict(rf.model, test)

# confusion matrix of the RF model
confusionMatrix(rf.preds, test$Class)

# variable importance
vip(rf.model)

## CLASSIFICATION TREE
# define the parameter grid
param.grid.ct <- expand.grid(cp = seq(0.01, 1, by = 0.01))

# control parameters
ctrl.ct <- trainControl(method = "cv", number = 5)

# select the best parameters for the CT
parameter.search.ct <- train(Class ~.,
                             data = train,
                             method = "rpart",
                             trControl = ctrl.ct,
                             tuneGrid = param.grid.ct)

# best parameter(s)
parameter.search.ct$bestTune

# building the model
ct.model <- rpart(Class~.,
                  data = train,
                  cp = parameter.search.ct$bestTune$cp)

# cross - validation error vs. CP
plotcp(ct.model)

# classification tree plot
rpart.plot(ct.model, digits = 3, box.palette = "pink")

# predictions of the CT model
ct.preds <- predict(ct.model, test, type = "class")

# confusion matrix of the CT
confusionMatrix(ct.preds, test$Class)

## ADA BOOSTING
# select the best parameters for the ADA BOOSTING
param.grid.ada <- expand.grid(iter = c(5, 10, 15, 20, 25),
                              maxdepth = c(1, 2, 3),
                              nu = seq(0.1, 1, by = 0.1))

# control parameters
ctrl.ada <- trainControl(method = "cv", number = 5)

# define the parameter grid
parameter.search.ada <- train(Class ~.,
                              data = train,
                              method = "ada",
                              trControl = ctrl.ada,
                              tuneGrid = param.grid.ada)

# building the model 
ada.model <- boosting(formula = Class~.,
                      data = train,
                      boos = TRUE,
                      nu = parameter.search.ada$bestTune$nu,
                      maxdepth = parameter.search.ada$bestTune$maxdepth,
                      iter = parameter.search.ada$bestTune$nu)

# predictions of the ADA BOOSTING model
ada.preds <- predict(ada.model, test)

# confusion matrix of the ADABOOSTING
confusionMatrix(as.factor(ada.preds$class), test$Class)

## KNN
# scaling only cols Area & Minor_Axis_Length
set.seed(101)
df.scaled <- df
df.scaled$Area <- (df.scaled$Area - min(df.scaled$Area)) / (max(df.scaled$Area) - min(df.scaled$Area))
df.scaled$Minor_Axis_Length <- (df.scaled$Minor_Axis_Length - min(df.scaled$Minor_Axis_Length)) / (max(df.scaled$Minor_Axis_Length) - min(df.scaled$Minor_Axis_Length))

# test & train split for the scaled df (df.scaled)
split.knn <- sample.split(df.scaled$Class, SplitRatio = 0.7)
test.knn <- subset(df.scaled, split.knn == FALSE)
train.knn <- subset(df.scaled, split.knn == TRUE)

# define the parameter grid
ctrl.knn <- trainControl(method = "cv", number = 5)

# define the parameter grid for the KNN model with k values from 1 to 10 
param.grid.knn <- expand.grid(k = 1:10)

# select the best parameters for the KNN model
parameter.search.knn <- train(x = train.knn [, -5],
                              y = train.knn [, 5],
                              method = "knn",
                              trControl = ctrl.knn,
                              tuneGrid = param.grid.knn)

# building the knn model
knn.model <- knn(train = train.knn [, -5],
                 test = test.knn [, -5] ,
                 cl = train.knn$Class,
                 k = 3)

# confusion matrix of the KNN model
confusionMatrix(knn.model, test.knn$Class)

## XGBOOST
# define the parameter grid
param.grid.xgb <- expand.grid(
  nrounds = c(5,50, 100, 200),
  max_depth = c(1, 2, 3),
  eta = seq(0.1, 0.3, by = 0.1),
  gamma = c(1, 2, 3),
  min_child_weight = 8,
  colsample_bytree = 0.8,
  subsample = 0.5
)

# control parameters
ctrl.xgb <- trainControl(method = "cv",
                         number = 5,
                         allowParallel = TRUE)

# select the best parameters for the xgb model
parameter.search.xgb <- train(x = train [, -5],
                              y = train [, 5],
                              trControl = ctrl.xgb,
                              tuneGrid = param.grid.xgb,
                              method = "xgbTree")

# writing out the optimum model
ctrl.xgb <- trainControl(method = "none",
                         allowParallel = TRUE)

# final grid
final.grid.xgb <- expand.grid(nrounds = parameter.search.xgb$bestTune$nrounds,
                              max_depth = parameter.search.xgb$bestTune$max_depth,
                              eta = parameter.search.xgb$bestTune$eta,
                              gamma = parameter.search.xgb$bestTune$gamma,
                              min_child_weight = parameter.search.xgb$bestTune$min_child_weight,
                              colsample_bytree = parameter.search.xgb$bestTune$colsample_bytree,
                              subsample = parameter.search.xgb$bestTune$subsample)

# building the model
xgb.model <- train(x = train [, -5],
                   y = train [, 5],
                   trControl = ctrl.xgb,
                   tuneGrid = final.grid.xgb,
                   verbose = TRUE,
                   method = "xgbTree")

# predictions of the XGB model
xgb.preds <- predict(xgb.model, test)

# confusion matrix of the XGB model
confusionMatrix(xgb.preds, test$Class)

## GRADIENT BOOSTING
# control parameters
ctrl.gbm <- trainControl(method = "cv", number = 5)

# define the parameter grid
param.grid.gbm <- expand.grid(n.trees = c(5, 20, 50, 100, 300),
                              shrinkage = c(0.01, 0.1, 0.3) ,
                              interaction.depth = c(1, 2, 3, 4),
                              n.minobsinnode = c(5, 10, 15, 20))

gbm.model <- train(Class ~.,
                   data = train,
                   method = "gbm",
                   trControl = ctrl.gbm,
                   tuneGrid = param.grid.gbm)

# predictions of the GBM model
gbm.preds <- predict(gbm.model, test)

# confusion matrix of the GBM model
confusionMatrix(gbm.preds, test$Class)