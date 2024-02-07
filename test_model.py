from darknet import *
import wget 
import os 

def get_test_input(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (416, 416))         
    img_ =  img[:,:,::-1].transpose((2,0,1))  
    img_ = img_[np.newaxis,:,:,:]/255.0       
    img_ = torch.from_numpy(img_).float()     
    img_ = Variable(img_)                  
    return img_

# Download file if doesn't exist
file_path = "dog-cycle-car.png"
if os.path.exists(file_path):
    "File alread present: Loading file"
else:
    "File not present: Downloading file"
    url = 'https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png'
    file_path = wget.download(url)

# Create network blocks
num_classes = 80
model = DarkNet("cfg/yolov3.cfg")

# Test model
inp = get_test_input(file_path)
pred = model(inp, False)
assert (pred.size() == torch.Size([1, 10647, 85]))
print(f"model outputs should be: [1, 10647, 85]. Got: {pred.shape} ")