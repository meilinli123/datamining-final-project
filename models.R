library(tidyverse)
library(ggplot2)
library(modelr)
library(rsample)
library(mosaic)
library(gamlr)
library(glmnet)
library(foreach)
library(caret)
library(nnet)
library(dplyr)
library(pROC)
library(performanceEstimation)
library(randomForest)
library(gbm)

data = read.csv("data/Bacteremia_public_S2.csv")
data$SEX <- ifelse(data$SEX == "1", 0, 1) 
data$BloodCulture <- ifelse(data$BloodCulture == "yes", 1, ifelse(data$BloodCulture == "no", 0, NA))

data <- data[, !names(data) %in% "ID"]
data = na.omit(data)

set.seed(1)
data_split =  initial_split(data, prop=0.8)
bact_train = training(data_split)
bact_test = testing(data_split)

#1. Lasso
bact_x <- scale(model.matrix(BloodCulture ~
                               AGE+PLT+MONO+SODIUM+MG+BUN+HS+GBIL+AP+GGT+LDH+CRP+EOSR+MONOR+NEUR+AGE*MG+AGE*EOSR+AGE*MONOR -1, data = bact_train))  
bact_y = bact_train$BloodCulture

bact_lasso = cv.glmnet(bact_x, bact_y, family = "binomial", weights = ifelse(bact_y == 1, 10, 1))

coefLasso = coef(bact_lasso) %>% 
  as.matrix() %>% 
  as.data.frame() 

bact_test_x <- scale(model.matrix(BloodCulture ~ AGE + PLT + MONO + SODIUM + MG + BUN + HS + GBIL + AP + GGT
                                  +LDH + CRP + EOSR + MONOR +
                                    NEUR+AGE*MG+AGE*EOSR+AGE*MONOR -1, data =
                                    bact_test))

phat_test <- predict(bact_lasso, newx = bact_test_x, s = "lambda.min", type = "response")
yhat_test <- ifelse(phat_test > 0.5, 1, 0)

bact_test$BloodCulture <- as.factor(bact_test$BloodCulture)

#The Caret package has a different definition of sensitivity. The TPR we use is TP/(TP+FN), which is noted as "specificity" in R.
confMatrix1 <- confusionMatrix(as.factor(yhat_test), bact_test$BloodCulture)
print(confMatrix1)
cat("Accuracy:", confMatrix1$overall['Accuracy'], "\n")
cat("TPR:", confMatrix1$byClass['Specificity'], "\n")

roc <- roc(bact_test$BloodCulture, phat_test)

auc_value <- auc(roc)
print(paste("AUC:", auc_value))

ci <- ci.auc(roc)
print(paste("AUC 95% CI:", ci[1], "-", ci[3]))

plot(roc, main="ROC Curve", col="#1c61b6", lwd=2)
abline(a=0, b=1, lty=2, col="gray")
text(0.8, 0.2, paste("AUC =", round(auc_value, 2)), border="black", cex=1.2, font=2)

#2. Logistic regression

bact_lm <- glm(BloodCulture ~ ., family=binomial,
               data=bact_train, weights = ifelse(bact_train$BloodCulture == 1, 10, 1))

phat_test <- predict(bact_lm, newdata = bact_test, type = "response")
yhat_test <- ifelse(phat_test > 0.5, 1, 0)

bact_test$BloodCulture <- as.factor(bact_test$BloodCulture)

confusionMatrix2 <- confusionMatrix(as.factor(yhat_test), bact_test$BloodCulture)
confusionMatrix2
cat("Accuracy:", confusionMatrix2$overall['Accuracy'], "\n")
cat("TPR:", confusionMatrix2$byClass['Specificity'], "\n")

roc <- roc(bact_test$BloodCulture, phat_test)

auc_value <- auc(roc)
print(paste("AUC:", auc_value))

ci <- ci.auc(roc)
print(paste("AUC 95% CI:", ci[1], "-", ci[3]))

plot(roc, main="ROC Curve", col="#1c61b6", lwd=2)
abline(a=0, b=1, lty=2, col="gray")
text(0.8, 0.2, paste("AUC =", round(auc_value, 2)), border="black", cex=1.2, font=2)


#3. Random forest

bact_train$BloodCulture <- as.factor(bact_train$BloodCulture)

bact_train_balanced <- smote(BloodCulture ~ AGE+PLT+MONO+SODIUM+MG+BUN+HS+GBIL+AP+GGT+LDH+CRP+EOSR+MONOR+NEUR+AGE*MG+AGE*EOSR+AGE*MONOR -1, bact_train, perc.over = 300, k = 10)

bact_rf = randomForest(BloodCulture ~ AGE+PLT+MONO+SODIUM+MG+BUN+HS+GBIL+AP+GGT+LDH+CRP+EOSR+MONOR+NEUR+AGE*MG+AGE*EOSR+AGE*MONOR -1, data = bact_train_balanced, ntree = 500, importance = TRUE)

predictions <- predict(bact_rf, bact_test, type = "prob")[,2]  
predictions_factor <- factor(ifelse(predictions > 0.5, "1", "0"), levels = c("0", "1"))
print(length(predictions_factor))
print(length(bact_test$BloodCulture))

bact_test$BloodCulture <- factor(bact_test$BloodCulture, levels = c("0", "1"))

conf_matrix3 <- confusionMatrix(predictions_factor, bact_test$BloodCulture)
print(conf_matrix3)
cat("Accuracy:", conf_matrix3$overall['Accuracy'], "\n")
cat("TPR:", conf_matrix3$byClass['Specificity'], "\n")  
roc_curve <- roc(response = bact_test$BloodCulture, predictor = predictions)
auc_value <- auc(roc_curve)
ci <- ci.auc(roc_curve)

print(paste("AUC:", auc_value))
print(paste("AUC 95% CI:", ci[1], "-", ci[3]))

plot(roc_curve, main="ROC Curve", col="#1c61b6", lwd=2)
abline(a = 0, b = 1, lty = 2, col = "gray")
text(0.8, 0.2, paste("AUC =", round(auc_value, 2)), border = "black", cex = 1.2, font = 2)


#4. PCA

bact_train_scaled <- scale(bact_train[, -which(names(bact_train) == "BloodCulture")])  
bact_test_scaled <- scale(bact_test[, -which(names(bact_test) == "BloodCulture")], center = attr(bact_train_scaled, "scaled:center"), scale = attr(bact_train_scaled, "scaled:scale"))

ggcorrplot::ggcorrplot(cor(bact_train_scaled), hc.order = TRUE)

pca_model <- prcomp(bact_train_scaled, center = TRUE, scale. = TRUE)

summary(pca_model)

bact_test_pca <- predict(pca_model, newdata = bact_test_scaled)

bact_train_pca <- predict(pca_model, newdata = bact_train_scaled)

var_explained <- cumsum(pca_model$sdev^2 / sum(pca_model$sdev^2))
num_components <- max(which(var_explained <= 0.9))
bact_train_pca <- bact_train_pca[, 1:num_components]
bact_test_pca <- bact_test_pca[, 1:num_components]


#5. PCA Logistic


bact_train_pca_15 <- bact_train_pca[, 1:15]
bact_test_pca_15 <- bact_test_pca[, 1:15]

bact_train_pca_15_df <- data.frame(bact_train_pca_15, BloodCulture = bact_train$BloodCulture)
bact_test_pca_15_df <- data.frame(bact_test_pca_15, BloodCulture = bact_test$BloodCulture)

bact_lm <- glm(BloodCulture ~ ., family=binomial,
               data=bact_train_pca_15_df, weights = ifelse(bact_train_pca_15_df$BloodCulture == 1, 10, 1))

phat_test <- predict(bact_lm, newdata = bact_test_pca_15_df, type = "response")
yhat_test <- ifelse(phat_test > 0.5, 1, 0)

bact_test_pca_15_df$BloodCulture <- as.factor(bact_test_pca_15_df$BloodCulture)

confusionMatrix <- confusionMatrix(as.factor(yhat_test), bact_test_pca_15_df$BloodCulture)
confusionMatrix
cat("Accuracy:", confusionMatrix$overall['Accuracy'], "\n")
cat("TPR:", confusionMatrix$byClass['Specificity'], "\n")

roc <- roc(bact_test_pca_15_df$BloodCulture, phat_test)

auc_value <- auc(roc)
print(paste("AUC:", auc_value))

ci <- ci.auc(roc)
print(paste("AUC 95% CI:", ci[1], "-", ci[3]))

plot(roc, main="ROC Curve", col="#1c61b6", lwd=2)
abline(a=0, b=1, lty=2, col="gray")
text(0.8, 0.2, paste("AUC =", round(auc_value, 2)), border="black", cex=1.2, font=2)


#6. PCA Lasso

bact_train_pca_15 <- bact_train_pca[, 1:15]
bact_test_pca_15 <- bact_test_pca[, 1:15]

bact_x_pca_15 <- as.matrix(bact_train_pca_15)
bact_y <- bact_train$BloodCulture 

bact_lasso_pca <- cv.glmnet(bact_x_pca_15, bact_y, family = "binomial", weights = ifelse(bact_y == 1, 10, 1))


coef_pca_15 <- coef(bact_lasso_pca, s = "lambda.min") %>%
  as.matrix() %>%
  as.data.frame()

bact_test_x_pca_15 <- as.matrix(bact_test_pca_15)

phat_test <- predict(bact_lasso, newx = bact_test_x, s = "lambda.min", type = "response")
yhat_test <- ifelse(phat_test > 0.5, 1, 0)

bact_test$BloodCulture <- as.factor(bact_test$BloodCulture)

confMatrix4 <- confusionMatrix(as.factor(yhat_test), bact_test$BloodCulture)
print(confMatrix4)
cat("Accuracy:", confMatrix4$overall['Accuracy'], "\n")
cat("TPR:", confMatrix4$byClass['Specificity'], "\n")

roc <- roc(bact_test$BloodCulture, phat_test)

auc_value <- auc(roc)
print(paste("AUC:", auc_value))

#7. PCA Random forest

bact_train_pca_15 <- bact_train_pca[, 1:15]
bact_test_pca_15 <- bact_test_pca[, 1:15]

bact_train_pca_15_df <- data.frame(bact_train_pca_15, BloodCulture = bact_train$BloodCulture)
bact_test_pca_15_df <- data.frame(bact_test_pca_15, BloodCulture = bact_test$BloodCulture)

bact_train_pca_15_df$BloodCulture <- as.factor(bact_train_pca_15_df$BloodCulture)

bact_train_balanced <- smote(BloodCulture ~ ., bact_train_pca_15_df, perc.over = 300, k = 10)

bact_rf = randomForest(BloodCulture ~ ., data = bact_train_balanced, ntree = 500, importance = TRUE, weights = ifelse(bact_train_balanced$BloodCulture == 1, 10, 1))

predictions_factor <- factor(ifelse(predictions > 0.5, "1", "0"), levels = c("0", "1"))

bact_test_pca_15_df$BloodCulture <- factor(bact_test_pca_15_df$BloodCulture, levels = c("0", "1"))

conf_matrix5 <- confusionMatrix(predictions_factor, bact_test_pca_15_df$BloodCulture)
print(conf_matrix5)
cat("Accuracy:", conf_matrix5$overall['Accuracy'], "\n")
cat("TPR:", conf_matrix5$byClass['Specificity'], "\n")  

roc_curve <- roc(response = bact_test_pca_15_df$BloodCulture, predictor = predictions)
auc_value <- auc(roc_curve)
ci <- ci.auc(roc_curve)

print(paste("AUC:", auc_value))
print(paste("AUC 95% CI:", ci[1], "-", ci[3]))

plot(roc_curve, main="ROC Curve", col="#1c61b6", lwd=2)
abline(a = 0, b = 1, lty = 2, col = "gray")
text(0.8, 0.2, paste("AUC =", round(auc_value, 2)), border = "black", cex = 1.2, font = 2)

roc_curve <- roc(response = bact_test_pca_15_df$BloodCulture, predictor = predictions)
auc_value <- auc(roc_curve)
ci <- ci.auc(roc_curve)

print(paste("AUC:", auc_value))
print(paste("AUC 95% CI:", ci[1], "-", ci[3]))

plot(roc_curve, main="ROC Curve", col="#1c61b6", lwd=2)
abline(a = 0, b = 1, lty = 2, col = "gray")
text(0.8, 0.2, paste("AUC =", round(auc_value, 2)), border = "black", cex = 1.2, font = 2)

#8. Boosting

bact_boost = gbm(BloodCulture ~ ., 
                 data = bact_train,
                 distribution = 'multinomial',  
                 interaction.depth = 4, 
                 n.trees = 1000, 
                 shrinkage = .05, weights = ifelse(bact_train$BloodCulture == 1, 10, 1))

predictions <- predict(bact_boost, newdata = bact_test, n.trees = 1000, type = "response")
predicted_classes <- apply(predictions, 1, which.max)  
predicted_classes <- factor(predicted_classes, levels = 1:nlevels(bact_test$BloodCulture), labels = levels(bact_test$BloodCulture))

bact_test$BloodCulture <- factor(bact_test$BloodCulture)

# Boosting Perfermance

confMatrix6 <- confusionMatrix(predicted_classes, bact_test$BloodCulture)
print(confMatrix6)
cat("Accuracy:", confMatrix6$overall['Accuracy'], "\n")
cat("TPR:", confMatrix6$byClass['Specificity'], "\n")  

#9. PCA Boosting

bact_train_pca_15_df <- data.frame(bact_train_pca_15, BloodCulture = bact_train$BloodCulture)
bact_test_pca_15_df <- data.frame(bact_test_pca_15, BloodCulture = bact_test$BloodCulture)

bact_train_pca_15_df$BloodCulture <- as.factor(bact_train_pca_15_df$BloodCulture)
bact_test_pca_15_df$BloodCulture <- as.factor(bact_test_pca_15_df$BloodCulture)

bact_boost = gbm(BloodCulture ~ ., 
                 data = bact_train_pca_15_df,
                 distribution = 'multinomial',  
                 interaction.depth = 10, 
                 n.trees = 500, 
                 shrinkage = .05,
                 verbose = FALSE, weights = ifelse(bact_train_pca_15_df$BloodCulture == 1, 10, 1))  

predictions <- predict(bact_boost, newdata = bact_test_pca_15_df, n.trees = 1000, type = "response")

predicted_classes <- apply(predictions, 1, which.max)
predicted_classes <- factor(predicted_classes, levels = 1:nlevels(bact_test_pca_15_df$BloodCulture), labels = levels(bact_test_pca_15_df$BloodCulture))

confMatrix7 <- confusionMatrix(predicted_classes, bact_test_pca_15_df$BloodCulture)
print(confMatrix7)
cat("Accuracy:", confMatrix7$overall['Accuracy'], "\n")
cat("TPR:", confMatrix7$byClass['Specificity'], "\n")  

