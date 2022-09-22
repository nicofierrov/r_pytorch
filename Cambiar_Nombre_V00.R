rm(list = ls())
graphics.off()
opar <- par()



#### PROBANDO PyTorch (Torch for R) y TensorFlow en R para usar Papers With Code.
#### https://paperswithcode.com/

# TORCH

#### https://www.rstudio.com/blog/torch/
#### https://torch.mlverse.org/



# TENSORFLOW --------------------------------------------------------------




# LIBRERIAS ---------------------------------------------------------------

library(tensorflow)
library(keras)


# TUTORIAL ----------------------------------------------------------------
# TUTORIAL: https://tensorflow.rstudio.com/tutorials/quickstart/beginner

### Load a dataset
c(c(x_train, y_train), c(x_test, y_test)) %<-% keras::dataset_mnist()

x_train <- x_train / 255
x_test <-  x_test / 255

### Build a machine learning model

model <- keras_model_sequential(input_shape = c(28, 28)) %>%
        layer_flatten() %>%
        layer_dense(128, activation = "relu") %>%
        layer_dropout(0.2) %>%
        layer_dense(10)

predictions <- predict(model, x_train[1:2, , ])
predictions


tf$nn$softmax(predictions)


loss_fn <- loss_sparse_categorical_crossentropy(from_logits = TRUE)


loss_fn(y_train[1:2], predictions)


model %>% compile(
        optimizer = "adam",
        loss = loss_fn,
        metrics = "accuracy"
)

### Train and evaluate your model

model %>% fit(x_train, y_train, epochs = 5)

model %>% evaluate(x_test,  y_test, verbose = 2)

probability_model <- keras_model_sequential() %>%
        model() %>%
        layer_activation_softmax() %>%
        layer_lambda(tf$argmax)

probability_model(x_test[1:5, , ])



# APLICAR UN MODELO DE PAPERS WITH CODE -----------------------------------


