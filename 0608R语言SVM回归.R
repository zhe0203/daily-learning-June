setwd("C:/Users/jk/Desktop/数据")
df=read.csv('container.csv')
mydata=df[,c('length','width','left','trail','draugth','Dwt')]
library(e1071)

# 将数据分为测试数据和训练数据
set.seed(234)
n = nrow(mydata)
indextrain = sample(1:n,round(3*n/4))
train = mydata[indextrain,]
test = mydata[-indextrain,]
predict_test = test[,1:5]

# svm
esp.svm.radopt = tune.svm(Dwt~.,data=train,cost=10^(-3:2),gamma=10^(-3:2),kernel="radial")

#根据cross validation 的结果来选择最优参数。之前两个参数的取值设定在10ˆ-3————10ˆ2之间
esp.svm.rad = svm(Dwt~.,data=train,
                  cost=esp.svm.radopt$best.parameters[1,2],
                  gamma=esp.svm.radopt$best.parameters[1,1]
                  ,kernel="radial")

# 计算训练数据集的拟合值
pred_train = fitted(esp.svm.rad)
# 将训练数据的拟合值与真实值组合成数据框
fit_train = data.frame(fit = pred_train,real = train$Dwt)
fit_train
# 预测
pred_test = predict(esp.svm.rad, predict_test)
# 将测试数据的拟合值与真实值组合成数据框
fit_test = data.frame(fit = pred_test,real = test$Dwt)
RMSE = sqrt(sum((fit_test$fit-fit_test$real)^2)/dim(test)[1])

# 对于新的数据进行预测
## 读取数据
newdata = read.csv('unique_static_2016.csv')
newdata = newdata[,c('length','width','LEFT','trail','draught')]
colnames(newdata)[3] = 'left'
colnames(newdata)[5] = 'draugth'
## 删除包含0的数据
result = apply(newdata,1,function(x) ifelse(any(x==0),1,0))
newdata$omit = result
## 删除
newdata = newdata[which(newdata$omit == 0),]
newdata = newdata[,1:5]

# 预测
pred_newdata = predict(esp.svm.rad,newdata)
# 合并数据
newdata$pred_Dwt = pred_newdata
write.csv(newdata,'预测值.csv')
