###############################
# Neural Nets with Keras
# Francisco Javier Carrera Arias
# 07/14/2017
###############################

library(keras)
library(corrplot)

# Sample Iris Data
dat <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"),
                header = FALSE)
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
set.seed(10)
mat <- as.matrix(scaled)
mat <- cbind(mat, dat[,5])
dimnames(mat) <- NULL
index <- sample(2,nrow(mat), replace = TRUE, prob = c(0.7, 0.3))
train <- mat[index == 1,1:4]
test <- mat[index == 2, 1:4]
targetTrain <- mat[index == 1, 5]
targetTest <- mat[index == 2, 5]

# Hot Encode labels
trainLabel <- to_categorical(targetTrain)
testLabel <- to_categorical(targetTest)
trainLabel <- trainLabel[,2:4]

model <- keras_model_sequential()
model %>%
  layer_dense(units = 8, activation = 'relu',
              input_shape = c(4)) %>%
  layer_dense(units = 3,
              activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'sgd',
  metrics = 'accuracy'
)

record <- model %>% fit(
  train,
  trainLabel,
  epochs = 1000,
  batch_size = 10,
  validation_split = 0.15
)

# Plot the history 
plot(record, bty = 'l')

score <- model %>% evaluate(
  test, 
  testLabel[,2:4],
  batch_size = 39
)
score
