import os
#from socket import CAN_RAW
import cv2

def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    car = os.listdir(dataPath + '/' + 'car')
    for i in car:
      img = cv2.imread(dataPath + '/' + 'car' + '/' + i, 0)
      img = cv2.resize(img, (36, 16))
      dataset.append((img, 1))
    
    car = os.listdir(dataPath + '/' + 'non-car')
    for i in car:
      img = cv2.imread(dataPath + '/' + 'non-car' + '/' + i, 0)
      img = cv2.resize(img, (36, 16))
      dataset.append((img, 0))
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    
    return dataset
