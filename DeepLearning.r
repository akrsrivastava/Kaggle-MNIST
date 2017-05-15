library(h2o)

h2o.init(ip="localhost",port=54321,nthreads = 4)

train <- h2o.uploadFile(path =  "train.csv" ,header=TRUE)
train$label <- as.factor(train$label)
test <- h2o.uploadFile("test.csv")
str(train)
dim(train)
head(train)
image(as.matrix(as.data.frame(train[1,-1]),nrow=28,ncol=28),axes=FALSE,col=grey(seq(0,1,length=256)))


mnist_model = h2o.deeplearning(x = 2:784, 
                               y = 1, 
                               training_frame = train,
                               activation= "RectifierWithDropout", 
                               hidden = c(100,100,100),
                               loss="Automatic",
                               distribution = "multinomial",
                               input_dropout_ratio = 0.2, 
                               #l1 = 1e-5, #validation = test_images.hex, 
                               epochs = 100)


predictions <- predict(mnist_model,test)
predictions <- as.data.frame(predictions)


response_df <- data.frame(seq(1:nrow(test)),predictions$predict)
colnames(response_df) <- c("ImageId","Label")
str(response_df)
write.csv(response_df,file="FirstSubmission_150517.csv",row.names=FALSE)
