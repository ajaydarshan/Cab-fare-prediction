rm(list=ls())
#Accept training and test data from command line
args = commandArgs(trailingOnly=TRUE)

# test if there are two argument: if not, return an error
if (length(args)<= 1) 
{
  stop("Two arguments must be supplied (train_data.csv and test_data.csv)", call.=FALSE)
}else if (length(args) == 2) 
{
  train_file = args[1]
  test_file = args[2]
}


load_libraries = c("DMwR","ggplot2","base","corrgram","stats","caret","randomForest","xgboost","gbm")

check_installed = function(x){
  if(!(x %in% row.names(installed.packages()))){
    install.packages(x)
  }
}
#install packages if not installed
lapply(load_libraries,check_installed)

#install.packages(load_libraries)

lapply(load_libraries,require,character.only = TRUE)

rm(load_libraries)
rm(check_installed)


#setwd("E:/Edwisor_project2/R")
wd = getwd()
if (!is.null(wd)) setwd(wd)


df = read.csv(train_file,header =TRUE,na.strings = c(""," ","NA"))
#Load test set and add same features as train set
test = read.csv(test_file,header =TRUE,na.strings = c(""," ","NA"))

test_pickup_datetime = test["pickup_datetime"]

head(df)

str(df)
summary(df)

#Exploratory data analysis
#remove rows having fractional passenger count and 0 count and greater than 10 passengers
#df = df[(df$passenger_count != 0),]
df = df[-which(df$passenger_count == 0 ),]
df = df[-which(df$passenger_count > 10),]
df = df[which(df$passenger_count%%1 == 0),]


#Convert fare_amount from factor to numeric
df$fare_amount = as.numeric(as.character(df$fare_amount))

#remove rows having fare amount -ve
df = df[-which(df$fare_amount <= 0),]

#Convert pickup_datetime from factor to date time
df$pickup_date = as.Date(as.character(df$pickup_datetime))
df$pickup_weekday = as.factor(format(df$pickup_date,"%u"))# Monday = 1
df$pickup_mnth = as.factor(format(df$pickup_date,"%m"))
df$pickup_yr = as.factor(format(df$pickup_date,"%Y"))
pickup_time = strptime(df$pickup_datetime,"%Y-%m-%d %H:%M:%S")
df$pickup_hour = as.factor(format(pickup_time,"%H"))

rm(pickup_time)

#Add same features to test set
test$pickup_date = as.Date(as.character(test$pickup_datetime))
test$pickup_weekday = as.factor(format(test$pickup_date,"%u"))# Monday = 1
test$pickup_mnth = as.factor(format(test$pickup_date,"%m"))
test$pickup_yr = as.factor(format(test$pickup_date,"%Y"))
pickup_time = strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test$pickup_hour = as.factor(format(pickup_time,"%H"))

rm(pickup_time)
 
#convert passenger_count to factor
df$passenger_count = as.factor(df$passenger_count)

df = subset(df,select = -c(pickup_datetime,pickup_date))
df = df[,c(2,3,4,5,6,7,8,9,10,1)]

#remove from test set
test = subset(test,select = -c(pickup_datetime,pickup_date))

################Missing value analysis
#numrows = nrow(df)
missing_vals = function (x)
{
  #(sum(is.na(x))/numrows) * 100 #missing percentage
  sum(is.na(x))
}

missing_values = data.frame(colnames(df),apply(df,MARGIN = 2,FUN = missing_vals),row.names = NULL)
names(missing_values) = c("Cols","Missing_values")
row.names(missing_values) = NULL
missing_values = missing_values[order(-missing_values$Missing_values),]

#df$fare_amount[100] = NA
#df$fare_amount[1] = NA
#df$fare_amount[is.na(df$fare_amount)] = median(df$fare_amount,na.rm = T)
#df$fare_amount = mean(df$fare_amount,na.rm = T)
df = knnImputation(df,k=2)

df = df[which(!is.na(df$pickup_mnth)),]
########Calculate the distance travelled using longitude and latitude
deg_to_rad = function(deg){
  (deg * pi) / 180
}


haversine = function(long1,lat1,long2,lat2){
  #long1rad = deg_to_rad(long1)
  phi1 = deg_to_rad(lat1)
  #long2rad = deg_to_rad(long2)
  phi2 = deg_to_rad(lat2)
  delphi = deg_to_rad(lat2 - lat1)
  dellamda = deg_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters

}

df$dist = haversine(df$pickup_longitude,df$pickup_latitude,df$dropoff_longitude,df$dropoff_latitude)

#Add to test set
test$dist = haversine(test$pickup_longitude,test$pickup_latitude,test$dropoff_longitude,test$dropoff_latitude)

#####Remove pickup and drop latitude and longitudes
df = subset(df,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
#remove from test set
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))

df = df[,c(1,2,3,4,5,7,6)]

#remove rows having distance zero, it is less likely to be considered as outlier
df = df[-which(df$dist == 0),]

########################Outlier analysis####################
pl1 = ggplot(df,aes(x = factor(passenger_count),y = fare_amount))
pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)

pl2 = ggplot(df,aes(x = factor(passenger_count),y = dist))
pl2 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)

# ## BoxPlots - Distribution and Outlier Check
numeric_index = sapply(df,is.numeric) #selecting only numeric

numeric_data = df[,numeric_index]

cnames = colnames(numeric_data)

#Imputation of outliers
for(i in cnames){
  vals = df[,i] %in% boxplot.stats(df[,i])$out
  df[which(vals),i] = NA
}

df = knnImputation(df,k=2)

################Feature selection###################

#Correlation analysis for numeric variables
corrgram(df[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")

#ANOVA for categorical variables with target numeric variable

#aov_results = aov(fare_amount ~ passenger_count * pickup_hour * pickup_weekday,data = df)
aov_results = aov(fare_amount ~ passenger_count + pickup_hour + pickup_weekday + pickup_mnth + pickup_yr,data = df)

summary(aov_results)

#pickup_weekday doesnt have much significance in determining fare_amount
df = subset(df,select=-pickup_weekday)

#remove from test set
test = subset(test,select=-pickup_weekday)

qqnorm(df$fare_amount)
histogram(df$fare_amount)


##############Sampling##################
set.seed(1000)
tr.idx = createDataPartition(df$fare_amount,p=0.8,list = FALSE)
train = df[tr.idx,]
test_data = df[-tr.idx,]

#train = df[1:12379,]
#test_data = df[12379:15473,]

###################Model Selection################
#Error metric used to select model is RMSE

#############Linear regression#################
lm_model = lm(fare_amount ~.,data=train)

summary(lm_model)

plot(lm_model$fitted.values,rstandard(lm_model),main = "Residual plot",
     xlab = "Predicted values of fare_amount",
     ylab = "standardized residuals")


lm_predictions = predict(lm_model,test_data[,1:5])

qplot(x = test_data[,6], y = lm_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,6],lm_predictions)
#mape is 17.9% , So accuracy is 82.1%
#rmse is 2.11 -> there is a difference of this much amount for predictions

#############Random forest#####################
rf_model = randomForest(fare_amount ~.,data=train)

summary(rf_model)

rf_predictions = predict(rf_model,test_data[,1:5])

qplot(x = test_data[,6], y = rf_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,6],rf_predictions)
#mape is 22.01% So accuracy is approx 78%
#rmse is 2.32

############GBM##########################
gbm_model = gbm(fare_amount ~.,data=train,n.trees = 400)

summary(gbm_model)

gbm_predictions = predict(gbm_model,test_data[,1:5],n.trees = 400)

qplot(x = test_data[,6], y = gbm_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,6],gbm_predictions)
#mape is 17.9% So accuracy is approx 82.1%
#rmse is 2.10

############XGBOOST###########################
train_data_matrix = as.matrix(sapply(train[-6],as.numeric))
test_data_data_matrix = as.matrix(sapply(test_data[-6],as.numeric))

xgboost_model = xgboost(data = train_data_matrix,label = train$fare_amount,nrounds = 15,verbose = FALSE)

summary(xgboost_model)
xgb_predictions = predict(xgboost_model,test_data_data_matrix)

qplot(x = test_data[,6], y = xgb_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,6],xgb_predictions)

#mape =17.64% So accuracy is 82.36%
#rmse = 2.08


#############Apply on test set####################
#As GBM and XGBoost gave better results. We will use these two models to predict test
###############GBM###########################
gbm_model2 = gbm(fare_amount ~.,data=df,n.trees = 400)

gbm_predictions2 = predict(gbm_model2,test,n.trees = 400)

gbm_pred_results = data.frame(test_pickup_datetime,"predictions" = gbm_predictions2)
write.csv(gbm_pred_results,"gbm_predictions_R.csv",row.names = FALSE)

###############XGBoost#######################
train_data_matrix2 = as.matrix(sapply(df[-6],as.numeric))
test_data_matrix2 = as.matrix(sapply(test,as.numeric))

xgboost_model2 = xgboost(data = train_data_matrix2,label = df$fare_amount,nrounds = 15,verbose = FALSE)

xgb_predictions2 = predict(xgboost_model2,test_data_matrix2)

xgb_pred_results = data.frame(test_pickup_datetime,"predictions" = xgb_predictions2)
write.csv(xgb_pred_results,"xgb_predictions_R.csv",row.names = FALSE)

