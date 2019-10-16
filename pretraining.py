

import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from scipy import signal
import sklearn
from PIL import Image

import tensorflow as tf

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=30, num_layers=3):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim  # 30 classes for starcraft
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Define LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)

        # Define output layer
        self.output = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we will initialize hidden states as.
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)

        h0 = Variable(h0)
        c0 = Variable(c0)

        return (h0, c0)

    def forward(self, x):
        # Forward pass
        # Create initial hidden and cell state.
        self.hidden = self.init_hidden()


        x = self.lstm(x, self.hidden)
        output = self.output(x)



# PyTorch Dataset to get data from tensorflow Dataset.
class TFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, session, num_samples):
        super(TFDataset, self).__init__()
        self.dataset = dataset
        self.session = session
        self.num_samples = num_samples
        self.next_element = None
        self.reset()

    def reset(self):
        dataset = self.dataset
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        return self

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        #print("getitem calledd!!!")

        session = self.session if self.session is not None else tf.Session()
        try:
            wav, label = session.run(self.next_element)
            x = self.log_spectrogram(wav.flatten(), sampling_rate=16000)
        except tf.errors.OutOfRangeError:
            self.reset()
            wav, label = session.run(self.next_element)

            x = self.log_spectrogram(wav.flatten(), sampling_rate=16000)
            #print("getitem calledd inside except!!!")
            #x_shape = x.shape
            #print(x.shape)
            #x = asarray(x)
            #pixels = x.astype('float32')
            #x = sklearn.preprocessing.normalize(x.flatten())
            #x=np.reshape(x,x_shape)
            #scaler = StandardScaler()


            #self.pop_mean.append(batch_mean)
            #self.pop_std.append(batch_std)
        print(x.shape)


        '''pop_mean = np.array(self.pop_mean).mean()
        pop_std = np.array(self.pop_std).mean()
        print(self.pop_mean)
        print(self.pop_std)
        
        #x = (x-pop_mean)/pop_std
        convert = tfms = transforms.Compose([       
                #transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=pop_mean, std=pop_std)
        ])
        
        print(x)

        for im in x:
            im = convert(im)

        #print(x.size)'''
        '''x_ar=[]
        for x_ in x:
            print(x_)
            #x_ = Image.fromarray(x_, mode="RGB")
            x_change = x_
            #x_change = convert(x_)
            x_change = (x_change-pop_mean)/pop_std
            print(x_change)
            x_ar.append(x_change)
        x = np.vstack(x_ar)'''
        ##return example.transpose(3, 0, 1, 2), label
        return x,np.argmax(label)

    def log_spectrogram (self,audio, sampling_rate, window_size=20,
                         step_size=10, eps=1e-10):

        nperseg = int(round(window_size * sampling_rate / 1000))

        # noverlap is the number of samples to overlap
        # between current window and next windows
        # 50% overlap (since step size=10, window_size=20,
        #                         step_size is 50% of window_size)
        # Example:step_size=10
        # noverlap = 10 * 44100/1000 = 441 (samples)
        noverlap = int(round(step_size * sampling_rate / 1000))
        freqs, times, specgram = signal.spectrogram(
            audio, sampling_rate, window='hann', nperseg=nperseg,
            noverlap=noverlap, detrend=False)

        # Note: We are adding eps to spectrogram
        # before performing log to avoid errors since log(0) is undefined
        # eps - epsilon - very small value close to 0
        #       here it is 1e-10 => 0.0000000001
        log_spec = np.log(specgram.T.astype(np.float32) + eps)
        log_spec = sklearn.preprocessing.normalize(log_spec,axis=0)
        image = Image.fromarray(log_spec.T, mode="RGB")
        #image.to(self.device)
        #print(image.size)
        convert = transforms.ToTensor()
        resize = transforms.Resize(24)
        tfms = transforms.Compose([
            transforms.Resize(24),
            #transforms.CenterCrop(24),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        '''image = resize(image)
        #image = convert(image)
        image = tfms(image)
        # image = convert(image)
        #print("tensor image size",image.shape)'''


        #print('Calculating mean and std of training dataset')
        #print(image.size)
        #image = resize(image)
        batch_mean = np.mean(image,axis=(0, 1, 2))
        batch_std = np.std(image,axis=(0, 1, 2))
        image = convert(image)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')

        image.to(device)

        #print("tensor image size",image.shape)

        image = (image-batch_mean)/batch_std


        return image#,batch_mean,batch_std



#dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self.num_parallel_readers)

# Start program
#if __name__ == "__main__":
#    import os
#    train_dataset = os.getcwd() + "/AutoDL_sample_data/starcraft/starcraft.data/train/sample-starcraft-train.tfrecord"
#    test_dataset = os.getcwd() + "/AutoDL_sample_data/starcraft/starcraft.data/test/sample-starcraft-test.tfrecord"
#
#    print(os.path.exists(train_dataset))
#    train_dataset = tf.data.TFRecordDataset(train_dataset)
#    test_dataset = tf.data.TFRecordDataset(test_dataset)
#    print(train_dataset)
#    print("#"*30)
#    print(test_dataset)
#    print("#"*30)




