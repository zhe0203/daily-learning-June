setwd("C:/Users/jk/Desktop/诸葛兼职/060811-200")
df = read.csv('wz.csv')
# 将数据转换为时间类型
df$timeOfMinPrice = as.POSIXlt(df$timeOfMinPrice,format='%H:%M:%S')
df$timeOfMaxPrice = as.POSIXlt(df$timeOfMaxPrice,format='%H:%M:%S')
# 事先确定好要划分的区间
time_1 = as.POSIXlt('9:28:00',format='%H:%M:%S')
time_2 = as.POSIXlt('10:00:00',format='%H:%M:%S')
time_3 = as.POSIXlt('10:30:00',format='%H:%M:%S')
time_4 = as.POSIXlt('11:00:00',format='%H:%M:%S')
time_5 = as.POSIXlt('11:30:00',format='%H:%M:%S')
time_6 = as.POSIXlt('13:00:00',format='%H:%M:%S')
time_7 = as.POSIXlt('13:30:00',format='%H:%M:%S')
time_8 = as.POSIXlt('14:00:00',format='%H:%M:%S')
time_9 = as.POSIXlt('14:30:00',format='%H:%M:%S')
time_10 = as.POSIXlt('15:00:00',format='%H:%M:%S')

# 生成新的数据集
min = c()
for (i in 1:nrow(df)){
  if (df[i,1] >= time_1 & df[i,1] < time_2){
    min = c(min,'[9:30,10:00)')
  } else if (df[i,1] >= time_2 & df[i,1] < time_3){
    min = c(min,'[10:00,10:30)')
  } else if (df[i,1] >= time_3 & df[i,1] < time_4){
    min = c(min,'[10:30,11:00)')
  } else if (df[i,1] >= time_4 & df[i,1] < time_5){
    min = c(min,'[11:00,11:30)')
  } else if (df[i,1] >= time_6 & df[i,1] < time_7){
    min = c(min,'[13:00,13:30)')
  } else if (df[i,1] >= time_7 & df[i,1] < time_8){
    min = c(min,'[13:30,14:00)')
  } else if (df[i,1] >= time_8 & df[i,1] < time_9){
    min = c(min,'[14:00,14:30)')
  } else if (df[i,1] >= time_9 & df[i,1] <= time_10){
    min = c(min,'[14:30,15:00)')
  }
}
df$min = min

# 生成新的数据集
min = c()
for (i in 1:nrow(df)){
  if (df[i,2] >= time_1 & df[i,2] < time_2){
    min = c(min,'[9:30,10:00)')
  } else if (df[i,2] >= time_2 & df[i,2] < time_3){
    min = c(min,'[10:00,10:30)')
  } else if (df[i,2] >= time_3 & df[i,2] < time_4){
    min = c(min,'[10:30,11:00)')
  } else if (df[i,2] >= time_4 & df[i,2] < time_5){
    min = c(min,'[11:00,11:30)')
  } else if (df[i,2] >= time_6 & df[i,2] < time_7){
    min = c(min,'[13:00,13:30)')
  } else if (df[i,2] >= time_7 & df[i,2] < time_8){
    min = c(min,'[13:30,14:00)')
  } else if (df[i,2] >= time_8 & df[i,2] < time_9){
    min = c(min,'[14:00,14:30)')
  } else if (df[i,2] >= time_9 & df[i,2] <= time_10){
    min = c(min,'[14:30,15:00)')
  }
}
df$max = min

# 将数据导出
write.csv(df,'result.csv')


# 数据转换
mydata = read.csv('xy.csv')
mydata$x = as.numeric(mydata$x)
mydata$y = as.numeric(mydata$y)
write.csv(mydata,'mydata_1.csv')
