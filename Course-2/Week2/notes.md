# Convolutional Neural Networks

## Augmentation: A technique to avoid overfitting

**Image augmentation and data augmentation** is one of the most widely used tools in deep learning to increase your dataset size and make your neural networks perform better.


For example, if we have a cat and our cats in our training dataset are always upright, we may not spot a cat that's lying down. But with augmentation, being able to rotate, skew, or making some other transforms to the image, we would be able to effectively generate that data to train off.


Augmentation simply amends your images on-the-fly while training using transforms like rotation. So, it could 'simulate' an image of a cat lying down by rotating a 'standing' cat by 90 degrees. As such you get a cheap way of extending your dataset beyond what you have already.


To learn more about Augmentation, and the available transforms, check out [keras - preprocessing](https://github.com/keras-team/keras-preprocessing). 
It's referred to as preprocessing for a very powerful reason: that it doesn't require you to edit your raw images, nor does it amend them for you on-disk. It does it in-memory as it's performing the training, allowing you to experiment without impacting your dataset.


### Coding augmentation with ImageDataGenerator

```python
train_datagen = ImageDataGenerator(rescale = 1/255,
                                   # Rotation range is a range from 0-180 degrees with which to randomly rotate images
                                   # So in this case, the image will rotate by random amount between 0 and 40 degrees
                                   rotation_range = 40,
                                   # Shifting, moves the image around inside its frame
                                   # Many pictures have the subject centered, so if we train based on those kind of images, we might over-fit for that scenario
                                   # These parameters specify, as a proportion of the image size, how much we should randomly move the subject around
                                   # So in this case, we might offset it by 20 percent vertically or horizontally
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   # This shears the image by random amounts up to the specified portion in the image
                                   # So in this case, it will shear up to 20 percent of the image
                                   shear_range = 0.2,
                                   # The 0.2 is a relative portion of the image you will zoom in on
                                   # So in this case, zooms will be a random amount up to 20 percent of the size of the image
                                   zoom_range = 0.2,
                                   # Kind of 'mirror effect' that helps us to flip the image
                                   horizontal_flip = True,
                                   # This fills in any pixels that might have been lost by the operations
                                   # In this case, 'nearest' uses neighbors of that pixel to try and keep uniformity
                                   fill_mode = 'nearest')

```

Image augmentation introduces a random element to the training images but if the validation set doesn't have the same randomness, then its results can fluctuate. So bear in mind that you don't just need a broad set of images for training, you also need them for testing or the image augmentation won't help you very much.