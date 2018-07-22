library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyr)
library(corrplot)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(vcd)

#Loading the data and doing some exploratory analysis
Hiring_Challenge <- read_csv("/Users/varshapandey/Downloads/Hiring_Challenge.csv")

data <- data.frame(Hiring_Challenge)
dim(data)
summary(data)
table(data$Hired)
data=data %>% mutate_if(is.character, as.factor)

str(data)
data=data %>% mutate_if(is.integer, as.numeric)

data$C2 <- as.numeric(data$C2)
data$C14 <- as.numeric(data$C14)
data$Hired <- as.factor(data$Hired)

cat_var <- list(names(data)[which(sapply(data, is.factor))])
cat_var
numeric_var <- names(data)[which(sapply(data, is.numeric))]
numeric_var

str(data)

#Univariate Analysis
lapply(data, function(x) {
  if (is.numeric(x)) return(summary(x))
  if (is.factor(x)) return(table(x))
})

ggplot(gather(data[,c(numeric_var)]),aes(value))+geom_histogram()+facet_wrap(~key, scales = 'free_x')


#Outlier Detection
boxplot(data$C2)
hist(data$C2)
boxplot(data$C3)
hist(data$C3) #Outliers
boxplot(data$C8) 
hist(data$C8) #Outliers
boxplot(data$C11)
hist(data$C11) #Outliers
boxplot(data$C14)
hist(data$C14)
boxplot(data$C15)
hist(data$C15) #Outliers

#Missing Value Detection
colSums(data=="?")
colSums(sapply(data, is.na))

#Bivariate Analysis
pairs(data[1:15], upper.panel = NULL)
lapply(data, function(x) (plot(x, data$Hired)))

#Correlation Analysis
variablesToKeep <- sapply(cat_var, paste0)
catcorrm <- function(cat_var, cat_data) sapply(cat_var, function(y) sapply(cat_var, function(x) assocstats(table(cat_data[,x], cat_data[,y]))$cramer))
catcorrm(variablesToKeep, data[,sapply(data,is.factor)])

plot(data$C4,data$C5)
chisq.test(data$C5,data$C4) # Highly correlated variables

variablesToDrop <- names(data) %in% c('C4')
data <- data[!variablesToDrop]

correlationMatrix <- cor(data[,sapply(data,is.numeric)])
corrplot(correlationMatrix, method = "number", type = "upper", tl.cex= 0.8)

#Training-Test Split
index <- sample(2, nrow(data),replace = T,prob = c(0.8,0.2))
data_train <- data.frame(data[index==1,])
data_test <- data.frame(data[index==2,])

str(data_train)
str(data_test)
table(data_train$Hired)

#Outlier Treatment for Training Data
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  caps <- quantile(x, probs=c(.05, .95), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- caps[1]
  y[x > (qnt[2] + H)] <- caps[2]
  y
}

boxplot.stats(data_train$C3)$out
boxplot(data_train$C3)
hist(data_train$C3)
data_train$C3 <- remove_outliers(data_train$C3)

boxplot.stats(data_train$C8)$out
boxplot(data_train$C8)
hist(data_train$C8)
data_train$C8 <- remove_outliers(data_train$C8)

boxplot.stats(data_train$C11)$out
boxplot(data_train$C11)
hist(data_train$C11)
data_train$C11 <- remove_outliers(data_train$C11)

boxplot.stats(data_train$C15)$out
boxplot(data_train$C15)
hist(data_train$C15)
data_train$C15 <- remove_outliers(data_train$C15)

#Fitting the model using Random Forest, KNN and Logistic Regression
set.seed(7)
fit.rf <- randomForest(Hired~ . ,data = data_train,mtry = 3,importance = TRUE)
fit.rf

fit.knn <- train(Hired~ . ,data = data_train,method="knn", metric="Accuracy", preProcess = c("center", "scale"),trControl = trainControl(method = "cv"))
fit.knn

fit.glm <- train(Hired ~ ., data=data_train, method="glm", metric="Accuracy", trControl=trainControl(method="cv",number=10))
fit.glm


#Transformation on Test Dataset
data_test$C3 <- remove_outliers(data_test$C3)
data_test$C8 <- remove_outliers(data_test$C8)
data_test$C11 <- remove_outliers(data_test$C11)
data_test$C15 <- remove_outliers(data_test$C15)

#Going with Random Forest with the best accuracy
data_test$pred_class <- predict(fit.rf,data_test)
pred_class <- predict(fit.rf,data_test)
confusionMatrix(pred_class,data_test$Hired,positive='1')
#Test set Accuracy of ~ 87.4%

predictions <- as.numeric(predict(fit.rf, data_test, type='response'))
multiclass.roc(data_test$Hired, predictions)
#ROC of ~86.9%

#Variable Importance
var.imp <- data.frame(importance(fit.rf,type=2))
var.imp$Variables <- row.names(var.imp)
var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),]
