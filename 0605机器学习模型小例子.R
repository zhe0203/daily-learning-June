# 读取数据
df = read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
colnames(df) = c('age','workclass','fnlwgt','enducation','educationnum',
                 'maritalstatus','occupation','relationship','race','sex',
                 'capitalgain','capitalloss','hoursperweek','nativecountry','salary')
df = df[,-5]

# 训练集和测试集划分
library(caret)
set.seed(123)
train_index <- createDataPartition(df$salary, p = .75, list = FALSE)
train_data = df[train_index,]    # 产生训练集
test_data = df[-train_index,]    # 产生测试集


# 决策树
library(rpart)
library(rpart.plot)
# 对决策树进行一些设置
ct = rpart.control(xval = 10,minsplit = 10)
# 建立决策树
fit <- rpart(salary~., data=train_data, method="class",control=ct)
# 绘制决策树图
rpart.plot(fit,type=2, extra=102,box.col="white",border.col="black",
           split.col="black",split.cex=1.2)
# 编写判断误判率的函数
count_result <- function(result,data_test){
  n = length(result)
  count_right = 0
  i = 1
  for (i in 1:n){
    if (result[i]==data_test[i,14]){
      count_right = count_right+1
    } 
  }
  return(count_right/n)  
}
result_test = predict(fit,test_data,type='class')   # 生成测试数据的预测值
result_train = predict(fit,train_data,type='class') # 生成训练数据的预测值
1 - count_result(result_test,test_data)    # 测试集误判率
1 - count_result(result_train,train_data)  # 训练集误判率

# 随机森林
library(randomForest)
rf <- randomForest(salary~., data=train_data, ntree=200,
                   importance=TRUE,proximity=FALSE)
# 计算测试数据集的误判率
test_forest = predict(rf,test_data)
test_table = table(test_forest,test_data$salary)
1 - sum(diag(test_table)/sum(test_table))
# 计算训练数据集的误判率
train_forest = predict(rf,train_data)
train_table = table(train_forest,train_data$salary)
1 - sum(diag(train_table)/sum(train_table))
# 查看每个变量在建模时的重要性
importance(rf)
varImpPlot(rf)

# 支持向量机SVM
library(e1071)
fit = svm(salary~.,data=train_data)
# 计算测试数据集的误判概率
test_svm = table(test_data$salary,predict(fit,test_data,type="class"))
1 - sum(diag(test_svm))/sum(test_svm)
# 计算测试训练集的误判概率
train_svm = table(train_data$salary,predict(fit,train_data,type="class"))
1 - sum(diag(train_svm))/sum(train_svm)
