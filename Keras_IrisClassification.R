###############################
# Neural Net with Keras
# Francisco Javier Carrera Arias
# 07/14/2017
###############################

library(keras)
library(tensorflow)
library(corrplot)

# Sample Iris Data
dat <- read.csv("Iris.csv")
str(dat)
names(dat) <- c('Sepal.length', 'Sepal.width', 'Petal.length', 'Petal.width', 'Species')

#Correlation Matrix
correlationMat <- cor(dat[,1:4])
corrplot(correlationMat, method = 'circle')

#Normalize Data
dat2 <- dat[,1:4]
maxs <- apply(dat2, 2, max)
mins <- apply(dat2, 2, min)
scaled <- as.data.frame(scale(dat2, center = mins, scale = maxs - mins))

# Train/Test Split
mat <- as.matrix(scaled)
mat <- cbind(mat, dat[,5])
dimnames(mat) <- NULL
index <- sample(2,nrow(mat), replace = TRUE, prob = c(0.8, 0.2))
train <- mat[index == 1,1:4]
test <- mat[index == 2, 1:4]
targetTrain <- mat[index == 1, 5]
targetTest <- mat[index == 2, 5]

# Hot Encode labels
trainLabel <- to_categorical(targetTrain)
testLabel <- to_categorical(targetTest)

model <- keras_model_sequential()
model %>%
  layer_dense(units = 16, activation = 'sigmoid',
              input_shape = c(4)) %>%
  layer_dense(units = 4, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

record <- model %>% fit(
  train,
  trainLabel,
  epochs = 500,
  batch_size = 8,
  validation_split = 0.15
)

score <- model %>% evaluate(
  test, 
  testLabel
)
score

predictions <- predict_classes(model,test)
tab <- table(targetTest,predictions)