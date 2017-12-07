#-----------------------------------Checkpoint - I - Data Analytics and Explorations-----------------------------------------------------------

install.packages('dplyr')
library(dplyr)
install.packages('missForest')
library(missForest)

#Load the Bank Data Set
#risk data
risk_data <- read.csv("D:/ML/ML Project/RISKFACTORSANDACCESSTOCARE.csv")

risk_data <- risk_data[c(1:7,10,13,16,19,22,25:31)]

# preventive service
preventive_data <- read.csv("D:/ML/ML Project/PreventiveService.csv")

preventive_data <- preventive_data[c(1:7,11,15,19,23,27,31,35:39)]

#VUNERABLE POPS AND ENV HEALTH
vulnerable_data <- read.csv("D:/ML/ML Project/VUNERABLEPOPSANDENVHEALTH.csv")

#MEASURESOFBIRTHANDDEATH

birthdeath_data <- read.csv("D:/ML/ML Project/MEASURESOFBIRTHANDDEATH.csv")
birthdeath_data <- birthdeath_data[c(1:6,19,49, 85, 91,97)]

#SUMMARYMEASURESOFHEALTH

health_measure_data <- read.csv("D:/ML/ML Project/SUMMARYMEASURESOFHEALTH.csv")
health_measure_data <- health_measure_data[c(1:7,11,17)]



library(plyr)

master_data <- join_all(list(risk_data,preventive_data,vulnerable_data,birthdeath_data,health_measure_data), 
                        by=c("State_FIPS_Code", "County_FIPS_Code","CHSI_County_Name","CHSI_State_Name",
                             "CHSI_State_Abbr","Strata_ID_Number"), type='left')

summary(master_data)

str(master_data)
sum(is.na(master_data$Recent_Drug_Use))

write.csv(master_data,file = "master.csv")

####--------DATA LOADING--------------------#####

master_data <- read.csv("master.csv")

master_data$HPSA_Ind <- as.factor(master_data$HPSA_Ind)
str(master_data$HPSA_Ind)
summary(master_data$HPSA_Ind)

master_data$No_Exercise[which(master_data$No_Exercise == -1111.1)] <- NA
master_data$Obesity[which(master_data$Obesity == -1111.1)] <- NA
master_data$High_Blood_Pres[which(master_data$High_Blood_Pres == -1111.1)] <- NA
master_data$Smoker[which(master_data$Smoker == -1111.1)] <- NA
master_data$Diabetes[which(master_data$Diabetes == -1111.1)] <- NA
master_data$Uninsured[which(master_data$Uninsured == -2222)] <- NA
master_data$Elderly_Medicare[which(master_data$Elderly_Medicare == -2222)] <- NA
master_data$Disabled_Medicare[which(master_data$Disabled_Medicare == -2222)] <- NA
master_data$Dentist_Rate[which(master_data$Dentist_Rate == -2222.2)] <- NA
master_data$Influenzae[which(master_data$Influenzae == -2224)] <- NA
master_data$HepA[which(master_data$HepA == -2224)] <- NA
master_data$HepB[which(master_data$HepB == -2224)] <- NA
master_data$Measeles[which(master_data$Measeles == -2224)] <- NA
master_data$Influenzae[which(master_data$Influenzae == -2224)] <- NA
master_data$Pertusis[which(master_data$Pertusis == -2224)] <- NA
master_data$Congential.Rubella[which(master_data$Congential.Rubella == -2224)] <- NA
master_data$Syphilis[which(master_data$Syphilis == -2224)] <- NA
master_data$Unemployed[which(master_data$Unemployed == -2222)] <- NA
master_data$Unemployed[which(master_data$Unemployed == -9999)] <- NA
master_data$Sev_Work_Disabled[which(master_data$Sev_Work_Disabled == -2222)] <- NA
master_data$Ecol[which(master_data$Ecol == -2224)] <- NA
master_data$Salmonella[which(master_data$Salmonella == -2224)] <- NA
master_data$Shig[which(master_data$Shig == -2224)] <- NA
master_data$Toxic_Chem[which(master_data$Toxic_Chem == -2222)] <- NA
master_data$Premature[which(master_data$Premature == -2222.2)] <- NA
master_data$Premature[which(master_data$Premature == -1111.1)] <- NA
master_data$Infant_Mortality[which(master_data$Infant_Mortality == -2222.2)] <- NA
master_data$Infant_Mortality[which(master_data$Infant_Mortality == -1111.1)] <- NA
master_data$Brst_Cancer[which(master_data$Brst_Cancer == -2222.2)] <- NA
master_data$Brst_Cancer[which(master_data$Brst_Cancer == -1111.1)] <- NA
master_data$Col_Cancer[which(master_data$Col_Cancer == -2222.2)] <- NA
master_data$Col_Cancer[which(master_data$Col_Cancer == -1111.1)] <- NA
master_data$CHD[which(master_data$CHD == -2222.2)] <- NA
master_data$CHD[which(master_data$CHD == -1111.1)] <- NA
master_data$All_Death[which(master_data$All_Death == -2222.2)] <- NA
master_data$All_Death[which(master_data$All_Death == -1111.1)] <- NA
master_data$Health_Status[which(master_data$Health_Status == -1111.1)] <- NA
master_data$ALE[which(master_data$ALE == -2222.2)] <- NA

#Master

master_data <- master_data[c(-1,-2,-3,-4,-6)]

state_labels <- master_data[1]
master_forest <- missForest(master_data[-1])

master_forest$OOBerror

master_data <- master_forest$ximp

master_data <- cbind(state_labels,master_data)


write.csv(master,file = "master_final1.csv")
