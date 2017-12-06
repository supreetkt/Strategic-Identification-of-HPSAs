
install.packages('Information')
##########--------------------############

master_data <- read.csv("master_final.csv")

summary(master_data)
str(master_data)

master_data$Community_Health_Center_Ind <- as.factor(master_data$Community_Health_Center_Ind)
master_data$Carbon_Monoxide_Ind <- as.factor(master_data$Carbon_Monoxide_Ind)
master_data$Ozone_Ind <- as.factor(master_data$Ozone_Ind) 
master_data$Particulate_Matter_Ind <- as.factor(master_data$Particulate_Matter_Ind)

iv_data <- master_data

master_data$HPSA_Ind <- as.factor(master_data$HPSA_Ind)
state_abbr <- model.matrix(~master_data$CHSI_State_Abbr)
state_abbr <- data.frame(state_abbr)
state_abbr <- state_abbr[,-1]

master_frame <- cbind(master_data[-1],state_abbr)


########------------------Outlier Treatment----------------#######

boxplot.stats(master_frame$No_Exercise)$out
quantile(master_frame$No_Exercise, seq(0,1,0.01))

quantile(master_frame$Obesity, seq(0,1,0.01))

quantile(master_frame$High_Blood_Pres, seq(0,1,0.01))

quantile(master_frame$Smoker, seq(0,1,0.01))

quantile(master_frame$Diabetes, seq(0,1,0.01))

quantile(master_frame$Uninsured, seq(0,1,0.01))

master_frame$Uninsured[which(master_frame$Uninsured> 152234.400)] <- 152234.400

quantile(master_frame$Elderly_Medicare, seq(0,1,0.01))

master_frame$Elderly_Medicare[which(master_frame$Elderly_Medicare> 24156.40)] <- 24156.40

quantile(master_frame$Disabled_Medicare, seq(0,1,0.01))

quantile(master_frame$Prim_Care_Phys_Rate, seq(0,1,0.01))

master_frame$Prim_Care_Phys_Rate[which(master_frame$Prim_Care_Phys_Rate> 207.24)] <- 207.24

quantile(master_frame$Dentist_Rate, seq(0,1,0.01))

master_frame$Dentist_Rate[which(master_frame$Dentist_Rate> 96.92)] <- 96.92

quantile(master_frame$Sev_Work_Disabled, seq(0,1,0.01))

master_frame$Sev_Work_Disabled[which(master_frame$Sev_Work_Disabled> 30906.600)] <- 30906.600

quantile(master_frame$ALE, seq(0,1,0.01))

master_frame <- master_frame[c(-31,-30)]

levels(master_frame$HPSA_Ind) <- c(0,1)



boxplot(master_frame$Prim_Care_Phys_Rate, col="royalblue2", xlab = "Primary Phy Rate")

boxplot(master_frame$Elderly_Medicare, xlab = "Elderly Medicare")

  ###----------Splitting the Data into Train and Test--------#######

library(caTools)
set.seed(100)
split_master_frame = sample.split(master_frame$HPSA_Ind, SplitRatio = 0.7)
table(split_master_frame)
train = master_frame[split_master_frame,]
test = master_frame[!(split_master_frame),]

summary(train$HPSA_Ind)
class(train$HPSA_Ind)
#--------------------Checkpoint IV - Modeling  --------------------------------------------#


model1 = glm(HPSA_Ind ~ ., data = train, family = "binomial")
summary(model1)
#The step function is executed with the primary model 
best_model = step(model1,direction = "both")

summary(best_model)

library(car)

vif(best_model)

model2 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured + Elderly_Medicare + 
                Disabled_Medicare + Prim_Care_Phys_Rate + Dentist_Rate + 
                Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + 
                master_data.CHSI_State_AbbrAL + master_data.CHSI_State_AbbrAR + 
                master_data.CHSI_State_AbbrCA + master_data.CHSI_State_AbbrFL + 
                master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                master_data.CHSI_State_AbbrID + master_data.CHSI_State_AbbrIL + 
                master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + 
                master_data.CHSI_State_AbbrMA + master_data.CHSI_State_AbbrMI + 
                master_data.CHSI_State_AbbrMN + master_data.CHSI_State_AbbrMO + 
                master_data.CHSI_State_AbbrMT + master_data.CHSI_State_AbbrNC + 
                master_data.CHSI_State_AbbrNE + master_data.CHSI_State_AbbrOH + 
                master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                master_data.CHSI_State_AbbrSC + master_data.CHSI_State_AbbrSD + 
                master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX + 
                master_data.CHSI_State_AbbrUT + master_data.CHSI_State_AbbrVA + 
                master_data.CHSI_State_AbbrVT + master_data.CHSI_State_AbbrWI + 
                master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
              family = "binomial", data = train)


summary(model2)
vif(model2)

model_3 <-  glm(formula = HPSA_Ind ~ Obesity + Uninsured + Elderly_Medicare + 
                  Disabled_Medicare + Prim_Care_Phys_Rate + Dentist_Rate + 
                  Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + 
                  master_data.CHSI_State_AbbrAL + master_data.CHSI_State_AbbrAR + 
                  master_data.CHSI_State_AbbrCA + master_data.CHSI_State_AbbrFL + 
                  master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                  master_data.CHSI_State_AbbrID + master_data.CHSI_State_AbbrIL + 
                  master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                  master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + 
                  master_data.CHSI_State_AbbrMA + master_data.CHSI_State_AbbrMI + 
                  master_data.CHSI_State_AbbrMN + master_data.CHSI_State_AbbrMO + 
                  master_data.CHSI_State_AbbrMT + master_data.CHSI_State_AbbrNC + 
                  master_data.CHSI_State_AbbrNE + master_data.CHSI_State_AbbrOH + 
                  master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                  master_data.CHSI_State_AbbrSC + master_data.CHSI_State_AbbrSD + 
                  master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX + 
                  master_data.CHSI_State_AbbrUT + master_data.CHSI_State_AbbrVA + 
                  master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
                family = "binomial", data = train)

summary(model_3)


vif(model_3)

model_4 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured + Elderly_Medicare + 
                 Disabled_Medicare + Prim_Care_Phys_Rate + Dentist_Rate + 
                 Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + 
                 master_data.CHSI_State_AbbrAL + master_data.CHSI_State_AbbrAR + 
                 master_data.CHSI_State_AbbrCA + master_data.CHSI_State_AbbrFL + 
                 master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                 master_data.CHSI_State_AbbrID + master_data.CHSI_State_AbbrIL + 
                 master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                 master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + 
                 master_data.CHSI_State_AbbrMA + master_data.CHSI_State_AbbrMI + 
                 master_data.CHSI_State_AbbrMN + master_data.CHSI_State_AbbrMO + 
                 master_data.CHSI_State_AbbrMT + master_data.CHSI_State_AbbrNC + 
                 master_data.CHSI_State_AbbrNE + master_data.CHSI_State_AbbrOH + 
                 master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                 master_data.CHSI_State_AbbrSC + master_data.CHSI_State_AbbrSD + 
                 master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX + 
                 master_data.CHSI_State_AbbrUT + master_data.CHSI_State_AbbrVA + 
                 master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
               family = "binomial", data = train)

summary(model_4)
vif(model_4)

model_5 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured  + 
                 Disabled_Medicare + Prim_Care_Phys_Rate + Dentist_Rate + 
                 Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + 
                 master_data.CHSI_State_AbbrAL + master_data.CHSI_State_AbbrAR + 
                 master_data.CHSI_State_AbbrCA + master_data.CHSI_State_AbbrFL + 
                 master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                 master_data.CHSI_State_AbbrID + master_data.CHSI_State_AbbrIL + 
                 master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                 master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + 
                 master_data.CHSI_State_AbbrMA + master_data.CHSI_State_AbbrMI + 
                 master_data.CHSI_State_AbbrMN + master_data.CHSI_State_AbbrMO + 
                 master_data.CHSI_State_AbbrMT + master_data.CHSI_State_AbbrNC + 
                 master_data.CHSI_State_AbbrNE + master_data.CHSI_State_AbbrOH + 
                 master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                 master_data.CHSI_State_AbbrSC + master_data.CHSI_State_AbbrSD + 
                 master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX + 
                 master_data.CHSI_State_AbbrUT + master_data.CHSI_State_AbbrVA + 
                 master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
               family = "binomial", data = train)

summary(model_5)

vif(model_5)


model_6 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured  + Prim_Care_Phys_Rate + Dentist_Rate + 
                 Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + 
                 master_data.CHSI_State_AbbrAL + master_data.CHSI_State_AbbrAR + 
                 master_data.CHSI_State_AbbrCA + master_data.CHSI_State_AbbrFL + 
                 master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                 master_data.CHSI_State_AbbrID + master_data.CHSI_State_AbbrIL + 
                 master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                 master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + 
                 master_data.CHSI_State_AbbrMA + master_data.CHSI_State_AbbrMI + 
                 master_data.CHSI_State_AbbrMN + master_data.CHSI_State_AbbrMO + 
                 master_data.CHSI_State_AbbrMT + master_data.CHSI_State_AbbrNC + 
                 master_data.CHSI_State_AbbrNE + master_data.CHSI_State_AbbrOH + 
                 master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                 master_data.CHSI_State_AbbrSC + master_data.CHSI_State_AbbrSD + 
                 master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX + 
                 master_data.CHSI_State_AbbrUT + master_data.CHSI_State_AbbrVA + 
                 master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
               family = "binomial", data = train)

summary(model_6)

vif(model_6)



model_7 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured  + Prim_Care_Phys_Rate + Dentist_Rate + 
                 Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + 
                 master_data.CHSI_State_AbbrAL + master_data.CHSI_State_AbbrAR + 
                 master_data.CHSI_State_AbbrCA + master_data.CHSI_State_AbbrFL + 
                 master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                 master_data.CHSI_State_AbbrID + master_data.CHSI_State_AbbrIL + 
                 master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                 master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + master_data.CHSI_State_AbbrMI + 
                 master_data.CHSI_State_AbbrMN + master_data.CHSI_State_AbbrMO + 
                 master_data.CHSI_State_AbbrMT + master_data.CHSI_State_AbbrNC + 
                 master_data.CHSI_State_AbbrNE + master_data.CHSI_State_AbbrOH + 
                 master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                 master_data.CHSI_State_AbbrSC + master_data.CHSI_State_AbbrSD + 
                 master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX + 
                 master_data.CHSI_State_AbbrUT + master_data.CHSI_State_AbbrVA + 
                 master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
               family = "binomial", data = train)

summary(model_7)
vif(model_7)


model_8 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured  + Prim_Care_Phys_Rate + Dentist_Rate + 
                 Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + 
                 master_data.CHSI_State_AbbrAL + master_data.CHSI_State_AbbrAR + 
                 master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                 master_data.CHSI_State_AbbrID + master_data.CHSI_State_AbbrIL + 
                 master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                 master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + master_data.CHSI_State_AbbrMI+ master_data.CHSI_State_AbbrMO + 
                 master_data.CHSI_State_AbbrMT + master_data.CHSI_State_AbbrNC + 
                 master_data.CHSI_State_AbbrNE + master_data.CHSI_State_AbbrOH + 
                 master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                 master_data.CHSI_State_AbbrSC + master_data.CHSI_State_AbbrSD + 
                 master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX + 
                 master_data.CHSI_State_AbbrUT + master_data.CHSI_State_AbbrVA + 
                 master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
               family = "binomial", data = train)

summary(model_8)
vif(model_8)


model_9 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured  + Prim_Care_Phys_Rate + Dentist_Rate + 
                 Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + master_data.CHSI_State_AbbrAR + 
                 master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + master_data.CHSI_State_AbbrIL + 
                 master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                 master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + master_data.CHSI_State_AbbrMO + 
                 master_data.CHSI_State_AbbrMT + master_data.CHSI_State_AbbrNC + 
                 master_data.CHSI_State_AbbrNE + master_data.CHSI_State_AbbrOH + 
                 master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                 master_data.CHSI_State_AbbrSC + master_data.CHSI_State_AbbrSD + 
                 master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX + 
                 master_data.CHSI_State_AbbrUT + master_data.CHSI_State_AbbrVA + 
                 master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
               family = "binomial", data = train)

summary(model_9)
vif(model_9)

model_10 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured  + Prim_Care_Phys_Rate + Dentist_Rate + 
                  Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + master_data.CHSI_State_AbbrAR + 
                  master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + master_data.CHSI_State_AbbrIL + 
                  master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                  master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + master_data.CHSI_State_AbbrMO + 
                  master_data.CHSI_State_AbbrNC + 
                  master_data.CHSI_State_AbbrNE + master_data.CHSI_State_AbbrOH + 
                  master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                  master_data.CHSI_State_AbbrSD + 
                  master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX 
                + master_data.CHSI_State_AbbrVA + 
                  master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
                family = "binomial", data = train)

summary(model_10)


model_11 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured  + Prim_Care_Phys_Rate + Dentist_Rate + 
                  Recent_Drug_Use + Ecol + Salmonella + Shig + Col_Cancer + master_data.CHSI_State_AbbrAR + 
                  master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                  master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                  master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + master_data.CHSI_State_AbbrMO + 
                  master_data.CHSI_State_AbbrNC + 
                  master_data.CHSI_State_AbbrNE + 
                  master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                  master_data.CHSI_State_AbbrSD + 
                  master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX 
                + master_data.CHSI_State_AbbrVA + 
                  master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
                family = "binomial", data = train)

summary(model_11)

model_12 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured  + Prim_Care_Phys_Rate + Dentist_Rate + 
                  Recent_Drug_Use + Ecol  + Shig + Col_Cancer + master_data.CHSI_State_AbbrAR + 
                  master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                  master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                  master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + master_data.CHSI_State_AbbrMO + 
                  master_data.CHSI_State_AbbrNC + 
                  master_data.CHSI_State_AbbrNE + 
                  master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                  master_data.CHSI_State_AbbrSD + 
                  master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX 
                + master_data.CHSI_State_AbbrVA + 
                  master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
                family = "binomial", data = train)

summary(model_12)
vif(model_12)

model_13 <- glm(formula = HPSA_Ind ~ Obesity + Uninsured  + Prim_Care_Phys_Rate + Dentist_Rate + 
                  Recent_Drug_Use + Ecol  + Shig  + master_data.CHSI_State_AbbrAR + 
                  master_data.CHSI_State_AbbrGA + master_data.CHSI_State_AbbrIA + 
                  master_data.CHSI_State_AbbrIN + master_data.CHSI_State_AbbrKS + 
                  master_data.CHSI_State_AbbrKY + master_data.CHSI_State_AbbrLA + master_data.CHSI_State_AbbrMO + 
                  master_data.CHSI_State_AbbrNC + 
                  master_data.CHSI_State_AbbrNE + 
                  master_data.CHSI_State_AbbrOK + master_data.CHSI_State_AbbrPA + 
                  master_data.CHSI_State_AbbrSD + 
                  master_data.CHSI_State_AbbrTN + master_data.CHSI_State_AbbrTX 
                + master_data.CHSI_State_AbbrVA + 
                  master_data.CHSI_State_AbbrWV + master_data.CHSI_State_AbbrWY, 
                family = "binomial", data = train)


# The VIF and the pvalues for all the variables in the model are satifactory to fix this model as a final model
summary(model_13)
library(car)
vif(model_13)

final_model <- model_13

# The performance for the train data is calculated 

train$predict_prob <- predict(final_model, type = "response")

test$predict_prob <- predict(final_model, newdata = test, type = "response")

library(ROCR)

model_score <- prediction(train$predict_prob, train$HPSA_Ind)

model_perf <- performance(model_score, "tpr", "fpr")

# The performance for the test data set is calculated and the ROC curve is calculated 
model_score_test <- prediction(test$predict_prob, test$HPSA_Ind)
model_perf_test <- performance(model_score_test, "tpr","fpr")

library(plotROC)

plot(model_perf_test,col = "red", lab = c(10,10,7)) 

ggplot(model_perf)+geom_line()

model_perf

# The Confusion matrix for the train and test data is calculated 

library(caret)

confusionMatrix(as.numeric(train$predict_prob > 0.6),train$HPSA_Ind, positive = "1")
confusionMatrix(as.numeric(test$predict_prob > 0.6),test$HPSA_Ind, positive = "1")

#The Area Under the Curve Value is calculated

auc<-performance(model_score,"auc")
auc




