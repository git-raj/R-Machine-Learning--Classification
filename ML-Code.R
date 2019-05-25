# KNN Classification of SMS Dataset
rm(list=ls()); cat("\014") # clear all
library(tm) # Use this package for Text Mining

load("Data/SMS_DTM.RData") # Load dtm from saved data

dtm <- as.matrix(dtm)
dtm <- dtm[1:1000,] # Subset DTM

# Split the Document-Term Matrix into Train & Test Datasets
library(class) # 
# Consider "spam" as Positive and "ham" as Negative
Positive <- "spam";
Negative <- "ham"; 
CM.Names <- c(Positive,Negative)

DS.Size <- dim(dtm)[1]
Test.Train.Percent <- 0.6 # Split Data into 60% for Training and 40% for Testing
ix.Range <- round(DS.Size*Test.Train.Percent)
train.Range <- seq(from = 1, to = ix.Range, by = 1);
test.Range <- seq(from = (ix.Range+1), to = DS.Size, by = 1)

train.doc <- dtm[train.Range,] # Dataset for which classification is already known
test.doc <- dtm[test.Range,] # Dataset we are trying to classify

# Generate TAGS - Correct answers for the Train dataset
z <- c(SMS.Data$Tag[train.Range])
sum(z==Positive) # 88 - Number of Positive events ("spam") in the Test Data
sum(z==Negative) # 512 - Number of Negative events ("ham") in the Test Data
z <- gsub(Positive,"Positive",z); z <- gsub(Negative,"Negative",z) # Replace Tags with 
Tags.Train <- factor(z, levels = c("Positive","Negative")) # kNN expects Tags to be Factors data type

# Generate TAGS - Correct answers for the Test dataset
z <- c(SMS.Data$Tag[test.Range]); 
sum(z==Positive) # 64 - Number of Positive events ("spam") in the Test Data
sum(z==Negative) # 336 - Number of Negative events ("ham") in the Test Data
z <- gsub(Positive,"Positive",z); z <- gsub(Negative,"Negative",z) # Replace Tags with 
Tags.Test <- factor(z, levels = c("Positive","Negative")) # kNN expects Tags to be Factors data type

# 1) KNN Classificationusing package "class" ====
set.seed(0)  
prob.test <- knn(train.doc, test.doc, Tags.Train, k = 2, prob=TRUE) # k-number of neighbors considered

### -----------
# Display Classification Results
a <- 1:length(prob.test)
b <- levels(prob.test)[prob.test] # asign your classification Tags (Positive or Negative)
c <- attributes(prob.test)$prob # asign your classification probability values 
d <- prob.test==Tags.Test # Logicaly match your classification Tags with the known "Tags"
result <- data.frame(Doc=a,Predict=b,Prob=c, Correct=d) # Tabulate your result
sum(d)/length(Tags.Test) # % Correct Classification (0.86)

# Insert your code to find: 
# Confusion Matrix 
auto_cm <- table(prob.test, Tags.Test)


TP <- auto_cm[1,1]
FP <- auto_cm[1,2]
FN <- auto_cm[2,1]
TN <- auto_cm[2,2]


#precision
precision <- TP/(TP + FP)
precision

#recall
recall <- TP/(TP + FN)
recall

#F score
f.score <- 2*((precision * recall)/(precision + recall))
f.score

##Accuracy
Accuracy <- (TP+TN)/(TP+TN+FP+FN)
Accuracy

#typed from slides pics 
#chapter8

#2

library(caret)

set.seed(0)

#prepare data by addiing to DTM a column of document's class
train.doc1 <- as.data.frame(train.doc, stringAsFactors=False);
train.doc1$doc.class <- as.character(Tags.Train)
test.doc1 <- as.data.frame(test.doc, stringAsFactors=False);
test.doc1$doc.class <- as.character(Tags.Test)
test.doc0001 <- as.factor(Tags.Test) #workaround

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

Knn.Train <- train(doc.class ~ ., data = train.doc1, method = "knn", trControl = ctrl)
Knn.Predict <- predict(Knn.Train, newdata = test.doc1)

CM.Knn <- confusionMatrix(Knn.Predict, test.doc0001)# test.doc1$doc.class)
FScore.Knn <- CM.Knn$byClass[7]
print(FScore.Knn)

knitr::kable(Knn.Train$results, digits = 6, format.args = list(big.mark = ","))
Knn.k <- Knn.Train$bestTune$k
Accuracy.Knn <- Knn.Train$results$Accuracy[Knn.Train$results$k == Knn.k]

#3 SVM classification
set.seed(100)
SVM.Train <- train(doc.class ~., data = train.doc1, method = "svmRadial", trControl = ctrl)
SVM.Predict <- predict(SVM.Train, newdata = test.doc1)

CM.SVM <- confusionMatrix(SVM.Predict, test.doc1$doc.class)
FScore.SVM <- CM.SVM$byClass[7]
print(FScore.SVM)
SVM.C <- SVM.Train$bestTune$C
Accuracy.SVM <- SVM.Train$results$Accuracy[SVM.Train$results$C == SVM.C]

#4 Decision Tree
set.seed(101)
Tree.Train <- train(doc.class ~ ., data = data.frame(train.doc1), method = "rpart", trControl = ctrl)
Tree.Predict <- predict(Tree.Train, newdata = data.frame(test.doc1))

CM.Tree <- confusionMatrix(Tree.Predict, test.doc1$doc.class)
FScore.Tree <- CM.Tree$byClass[7]
Tree.cp <- Tree.Train$bestTune$cp
Accuracy.Tree <- Tree.Train$results$Accuracy[Tree.Train$results$cp ==Tree.cp]

