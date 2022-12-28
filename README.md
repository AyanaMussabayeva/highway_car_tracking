# highway_car_tracking

## Input data 
- Static video of highway 'highway.mp4' should be stored in the 'highway_car_tracking/data/' folder


## Pipeline description 

### Frame generator 
- Sampling video into [680x340] frames with sample_rate = 2

### Background removal 
- For each batch of images background is generated as median value of every pixel in each image of the corresponding batch
![Alt text](/misc/background.png "Background example") 

### Morphological operations
- Erode with small kernel for noise removal. 
- Dilatation with large kernel for connecting splitted by mistake contours

### Binarization and contour analysis 
- Custom thresholding for binarization 
- Outer contours selection with area > 500

### Tracking 
- Linking bounding boxes from neighbouring frames by IoU > 0.05 



