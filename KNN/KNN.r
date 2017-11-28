
# K-NN Model:

# master.knn
master_data <- read.csv("master_final1.csv")
# Bring the data in the correct format to implement K-NN model


state_abbr <- model.matrix(~master_data$CHSI_State_Abbr)
state_abbr <- data.frame(state_abbr)
state_abbr <- state_abbr[,-1]

master_frame <- cbind(master_data[-1],state_abbr)

str(master_frame)

# Convert factor columns to numeric 
master_frame$Community_Health_Center_Ind <- as.numeric(master_frame$Community_Health_Center_Ind)
master_frame$Carbon_Monoxide_Ind <- as.numeric(master_frame$Carbon_Monoxide_Ind)
master_frame$Ozone_Ind <- as.numeric(master_frame$Ozone_Ind)
master_frame$Particulate_Matter_Ind <- as.numeric(master_frame$Particulate_Matter_Ind)

master_frame <- master_frame[c(12,1:11,13:91)]

master_frame$HPSA_Ind <- as.factor(master_frame$HPSA_Ind)

# Split dataset into train and test dataset  
set.seed(100)
indices=sample(1:nrow(master_frame),0.7*nrow(master_frame))

train.knn=master_frame[indices,]
test.knn=master_frame[-indices,]


# Implement the K-NN model for optimal K.

# True class labels of train data
cl.train.knn=train.knn[,1]

# True class labels of test data
cl.test.knn=test.knn[,1]

# Cross Validation to find optimal k
model.knn=train(HPSA_Ind~.,data=train.knn,method='knn',tuneGrid=expand.grid(.k=1:50),
                metric='Accuracy',trControl=trainControl(method = 'repeatedcv',number = 10,
                                                         repeats=15))

# Plot to get optimal K  
plot(model.knn)
# We get optimal K as 22

library('class')  

# KNN - 29 Nearest neighbours
impknn22 <- knn(train.knn,test.knn, cl.train.knn, k = 22,prob = TRUE)
table(impknn22,test.knn[,1])
confusionMatrix(impknn22, test.knn[,1], positive ="2")

# Convert probabilities to a single class
attr(impknn22,"prob") <- ifelse(impknn22==2,attr(impknn22,"prob"),1 - attr(impknn22,"prob"))

# Plot ROC 
pred.knn=prediction(attr(impknn22, "prob"),test.knn[,1])
perf.knn=performance(pred.knn,"tpr","fpr")
plot(perf.knn,col='red',lty=2,lwd=2)

# Area under Curve    
auc.knn=performance(pred.knn,"auc")    
auc.knn


