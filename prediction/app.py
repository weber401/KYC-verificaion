#1
#import libraries

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from flask import Flask, app, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)


#2
#import libraries
import torch
from torch.autograd import Variable
import time
import os
import sys
import os
from torch import nn
from torchvision import models

#3
#Model with feature visualization
from torch import nn
from torchvision import models
class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))
    
#4
im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png',image*255)
    return image

def predict(model,img,last_file, path = './'):
  fmap,logits = model(img.to('cpu'))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _,prediction = torch.max(logits,1)
  confidence = logits[:,int(prediction.item())].item()*100
  print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:])
  result = heatmap * 0.5 + img*0.8*255
  cv2.imwrite('/content/1.png',result)

    # to save images in /outputimage
  output_dir = os.path.join(path, 'outputimage')
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, last_file)
  cv2.imwrite(output_path, result)

  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((r,g,b))
#   this pops up image
#   plt.imshow(result1)
#   plt.show()
  return [int(prediction.item()),confidence]
#img = train_data[100][0].unsqueeze(0)
#predict(model,img)

#5
#!pip3 install face_recognition
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length = 60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)      
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            faces = face_recognition.face_locations(frame)
            try:
              top,right,bottom,left = faces[0]
              frame = frame[top:bottom,left:right,:]
            except:
              pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
              break
        #print("no of frames",len(frames))
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image
def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype(int))
    plt.show()

import glob

#6
#Code for making prediction
def get_prediction(path_to_videos):
    im_size = 112
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((im_size,im_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)])

    

    video_dataset = validation_dataset(path_to_videos,sequence_length = 20,transform = train_transforms)
    model = Model(2)
    path_to_model = glob.glob('model\checkpoint.pt')[0]

    checkpoint = torch.load(path_to_model, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    for i in range(0,len(path_to_videos)):
        print(path_to_videos[i])

        # to get the name oft the video
        video_name = os.path.splitext(os.path.basename(path_to_videos[i]))[0]  # Get the video name without extension
        last_file = video_name + ".png"# Set last_file to the desired output path
        print("last_file",last_file)

        prediction = predict(model,video_dataset[i], last_file, './')
        if prediction[0] == 1:
            print("REAL")
            return "True"
        else:
            print("FAKE")
            return "False"

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

app.config['UPLOAD_FOLDER'] = 'C:/Users/dipen/OneDrive/Desktop/final-year-project/prediction/input_videos'

@app.route('/predict-video', methods=['POST'])
def predict0():
    if 'video' not in request.files:
        print (request.files)
        return jsonify({'error': 'no file found'})
    
    file = request.files['video'].read()
    # save the uploaded video file to a directory
    file_name = secure_filename(request.files['video'].filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    print(file_path)
    with open(file_path, 'wb') as f:
        f.write(file)

    path_to_videos = glob.glob(file_path)
    prediction = get_prediction(path_to_videos)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
