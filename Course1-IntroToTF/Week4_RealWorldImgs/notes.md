# Using Real-world Images :framed_picture:


What happens when you use larger images and where the feature might be in different locations?


The earlier examples with a fashion data used a built-in dataset. All of the data was handily split into training and test sets for you and labels were available. In many scenarios, that's not going to be the case and you'll have to do it for yourself.


One feature of the image generator in TensorFlow is that you can point it at a directory and then the sub-directories of that will automatically generate labels for you.

Directory structure:

![alt text](https://github.com/Immich/TensorFlowEspecialization/blob/master/Course-1/imgs/imageGenerator.png "TensorFlow Image Generator")

```python
# We pass rescale to it to normalize the data.
train_datagen = ImageDataGenerator(rescale 1./255)

# Call the flow from directory method on it to get it to load images from that directory and its sub-directories
# You should always point it at the directory that contains sub-directories that contain your images
# The names of the sub-directories will be the labels for your images that are contained within them
train_generator = train_datagen.flow_from_directory(
  train_dir,
  # The images will need to be resized to make them consistent
  target_size = (300, 300),
  # The images will be loaded for training and validation in batches where it's more efficient than doing it one by one
  batch_size = 128,
  class_mode = 'binary'
)
```


## Training the ConvNet with fit_generator

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss = 'binary_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['acc'])
              

# Training
history = model.fit_generator(
  # The first parameter is the training generator that you set up earlier. This streams the images from the training directory
  train_generator,
  # There are 1,024 images in the training directory, so we're loading them in 128 at a time. So in order to load them all, we need to do 8 batches
  steps_per_epoch = 8,
  epochs = 15,
  # Specify the validation set that comes from the validation_generator that we created earlier
  validation_data = validation_generator,
  # It had 256 images, and we wanted to handle them in batches of 32, so we will do 8 steps
  validation_steps = 8,
  # Specifies how much to display while training is going on. With verbose set to 2, we'll get a little less animation hiding the epoch progress.
  verbose = 2)
```




