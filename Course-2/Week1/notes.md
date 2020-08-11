# Convolutional Neural Networks

## Exploring a Larger Dataset


With a smaller dataset, you are at great risk of overfitting; with a larger dataset, then you have less risk of over-fitting, but overfitting can still happen.

```python
from tensorflow.keras.preprocessing.image
import DataGenerator

# To use an image generator, you should create an instance of one. If the data isn't already normalized, you can do that with the rescale parameter
train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# Call the flow from directory to get a generator object
train_generator = train_datagen.flow_from_directory(
                                                    # Point at the training directory for the training dataset
                                                    train_dir,
                                                    # In this case, the images are an all shapes and sizes, so we will resize them to 150 by 150
                                                    target_size = (150, 150),
                                                    # There's 2,000 images, so we'll use a 100 batches of 20 each
                                                    batch_size = 20,
                                                    # There are two classes that we want to classify so we set a binary class_mode
                                                    class_mode = 'binary')
    
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))
    
```

Now in Compilation, remember you can tweak the learning rate by adjusting the lr parameter.

```python
from tensorflow.keras.optimizers import RMSprop
model.compile(loss = 'binary_cross_entropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['acc'])
```


So now to train, we can call `model.fit` generator and pass it the training generator and the validation generator.

```python
history = model.fit_generator(
                              train_generator,
                              steps_per_epoch = 100,
                              epochs = 15,
                              validation_data = validation_generator,
                              validation_steps = 50,
                              verbose = 2)
```
