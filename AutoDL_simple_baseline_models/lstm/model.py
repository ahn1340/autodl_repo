# Modified by: Shangeth Rajaa, ZhengYing, Isabelle Guyon

"""An example of code submission for the AutoDL challenge in PyTorch.

It implements 3 compulsory methods: __init__, train, and test.
model.py follows the template of the abstract class algorithm.py found
in folder AutoDL_ingestion_program/.

The dataset is in TFRecords and Tensorflow is used to read TFRecords and get the 
Numpy array which can be used in PyTorch to convert it into Torch Tensor.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights. The final zip file should
not exceed 300MB.
"""



"""
Search for '# PYTORCH' to get directly to PyTorch Code.
"""



import torch.utils.data as data_utils
import torch
import tensorflow as tf
import os
import numpy as np

# Import the challenge algorithm (model) API from algorithm.py
import algorithm

# Other useful modules
import datetime
import time
np.random.seed(42)

# Torch import
import torch.nn as nn
from torch.autograd import Variable

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm

# Feature extraction
from librosa.feature import mfcc
import sklearn.preprocessing

# PYTORCH
# Make pytorch model in torchModel class
class torchModel(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(torchModel, self).__init__()
    self.fc1 = nn.Linear(input_dim, int((input_dim+output_dim)/2))
    self.fc2 = nn.Linear(int((input_dim+output_dim)/2), output_dim)
    self.dropout = nn.Dropout(0.3)
    self.relu = nn.ReLU()
    self.log_softmax = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = x.contiguous().view(x.size(0), -1)
    x = self.fc1(x)
    x = self.dropout(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.log_softmax(x)
    return x


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=30, num_layers=10):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim  # 30 classes for starcraft
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define output layer
        self.output = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self, batch_size):
        # This is what we will initialize hidden states as.
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        h0 = Variable(h0)
        c0 = Variable(c0)

        return (h0, c0)

    def forward(self, x):
        # Forward pass
        # Get the batch size for initializing hidden state
        batch_size = x.shape[0]
        # Create initial hidden and cell state.
        self.hidden = self.init_hidden(batch_size)


        x, hidden = self.lstm(x, self.hidden)
        output = self.output(x[:, -1, :])
        return output



class Model(algorithm.Algorithm):


  def __init__(self, metadata):
    super(Model, self).__init__(metadata)
    self.no_more_training = False
    self.output_dim = self.metadata_.get_output_size()
    self.num_examples_train = self.metadata_.size()
    row_count, col_count = self.metadata_.get_matrix_size(0)
    channel = self.metadata_.get_num_channels(0)
    sequence_size = self.metadata_.get_sequence_size()
    


    # Input dimension of the raw data? TODO:should it be feature dimension?
    self.input_dim = (13, 32)
    self.input_dim = 1500


    # Attributes for preprocessing
    self.default_shuffle_buffer = 100


    # getting an object for the PyTorch Model class for Model Class
    # use CUDA if available
    self.pytorchmodel = torchModel(self.input_dim, self.output_dim)
    self.pytorchmodel = LSTM(13, 100)
    #if torch.cuda.is_available(): self.pytorchmodel.cuda()


     # Attributes for managing time budget
    # Cumulated number of training steps
    self.birthday = time.time()
    self.total_train_time = 0
    self.cumulated_num_steps = 0
    self.estimated_time_per_step = None
    self.total_test_time = 0
    self.cumulated_num_tests = 0
    self.estimated_time_test = None
    self.trained = False
    self.done_training = False

    # PYTORCH
    # Critical number for early stopping
    self.num_epochs_we_want_to_train = 1

    # no of examples at each step/batch
    self.batch_size = 64

    


  def preprocess_tensor_4d(self, tensor_4d):
    """Preprocess a 4-D tensor (only when some dimensions are `None`, i.e.
    non-fixed). The output tensor wil have fixed, known shape.
    Args:
      tensor_4d: A Tensor of shape
          [sequence_size, row_count, col_count, num_channels]
          where some dimensions might be `None`.
    Returns:
      A 4-D Tensor with fixed, known shape.
    """
    tensor_4d_shape = tensor_4d.shape
    print_log("Tensor shape before preprocessing: {}".format(tensor_4d_shape))

    #if tensor_4d_shape[0] > 0 and tensor_4d_shape[0] < 10:
    #  num_frames = tensor_4d_shape[0]
    #else:
    #  num_frames = self.default_num_frames
    num_frames = 16000 #TODO: make this variable length
    if tensor_4d_shape[1] > 0:
      new_row_count = tensor_4d_shape[1]
    else:
      new_row_count=self.default_image_size[0]
    if tensor_4d_shape[2] > 0:
      new_col_count = tensor_4d_shape[2]
    else:
      new_col_count=self.default_image_size[1]

    if not tensor_4d_shape[0] > 0:
      print_log("Detected that examples have variable sequence_size, will " +
                "randomly crop a sequence with num_frames = " +
                "{}".format(num_frames))
      tensor_4d = crop_time_axis(tensor_4d, num_frames=num_frames)
    if not tensor_4d_shape[1] > 0 or not tensor_4d_shape[2] > 0:
      print_log("Detected that examples have variable space size, will " +
                "resize space axes to (new_row_count, new_col_count) = " +
                "{}".format((new_row_count, new_col_count)))
      tensor_4d = resize_space_axes(tensor_4d,
                                    new_row_count=new_row_count,
                                    new_col_count=new_col_count)

    # Audio signals are 1-D
    tensor_4d = tf.squeeze(tensor_4d)
    print_log("Tensor shape after preprocessing: {}".format(tensor_4d.shape))

    return tensor_4d



  def get_dataloader(self, tfdataset, batch_size, train=False):
    '''
    # PYTORCH
    This function takes a tensorflow dataset class and comvert it into a 
    Pytorch Dataloader of batchsize.
    This function is usually used for testing data alone, as training data
    is huge and training is done step/batch wise, rather than epochs.
    '''
    tfdataset = tfdataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))
    iterator = tfdataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    features = []
    labels = []
    if train:
      while True:
        try:
          x,y = sess.run(next_element)
          features.append(x)
          labels.append(y)
        except tf.errors.OutOfRangeError:
          break
      features = np.vstack(features)
      features = torch.Tensor(features)
      labels = torch.Tensor(labels)
      dataset = data_utils.TensorDataset(features, labels)
      loader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    else:
      while True:
        try:
          x , _= sess.run(next_element)
          x = mfcc(x, sr=16000, n_mfcc=13)
          features.append(x)
        except tf.errors.OutOfRangeError:
          break
      features = np.stack(features)
      features = np.transpose(features, (0, 2, 1))
      features = torch.Tensor(features)
      print("features shape: ", features.shape)
      dataset = data_utils.TensorDataset(features)
      loader = data_utils.DataLoader(dataset, batch_size=self.batch_size)
    return loader


  def input_function(self, dataset, is_training):
    """Given `dataset` received by the method `self.train` or `self.test`,
    prepare input to feed to model function.

    For more information on how to write an input function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_an_input_function
    
    # PYTORCH
    This function returns a tensorflow data iterator which is then converted to 
    PyTorch Tensors.
    """

    dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))

    if is_training:
      # Shuffle input examples
      dataset = dataset.shuffle(buffer_size=self.default_shuffle_buffer)
      # Convert to RepeatDataset to train for several epochs
      dataset = dataset.repeat()

    # Set batch size
    dataset = dataset.batch(batch_size=self.batch_size)

    iterator_name = 'train_iterator' if is_training else 'iterator_test'

    if not hasattr(self, iterator_name):
      self.train_iterator = dataset.make_one_shot_iterator()
    iterator = self.train_iterator
    return iterator




  def get_torch_tensors(self):
    '''
    # PYTORCH
    This function returns X and y Torch tensors from the tensorflow
    data iterator.
    X is transposed as images need specific dimensions in PyTorch.
    y is converted to single integer value from One-hot vectors.
    '''
    next_iter = self.training_data_iterator.get_next()
    sess = tf.Session()
    wavs, labels = sess.run(next_iter)
    features = []
    for i in range(self.batch_size):
      feature = mfcc(wavs[i], sr=16000, n_mfcc=13)
      # Normalize mfcc features along each channel
      feature = sklearn.preprocessing.normalize(feature)
      features.append(feature)

      #mfcc_data = np.swapaxes(feature, 0, 1)
      #fig, ax = plt.subplots()
      #cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
      #ax.set_title('MFCC')
      #ax.set_xlabel("Features")
      #ax.set_ylabel("Time")
      ## Showing mfcc_data
      #plt.show()
      #plt.plot(wavs[i])
      #plt.show()

    # shape = (batch, feature, seq_len)
    features = np.stack(features)
    # reshape to (batch, seq_len, feature)
    features = np.transpose(features, (0, 2, 1))

    #print(images.shape)
    #print(images[0])
    #plt.plot(images[0])
    #plt.show()
    return features, labels  # for multilabel it will be multi-hot vectors


    
    


  def trainloop(self, criterion, optimizer, dataset, steps):
    '''
    # PYTORCH
    Trainloop function does the actual training of the model
    1) it gets the X, y from tensorflow dataset.
    2) convert X, y to CUDA
    3) trains the model with the Tesors for given no of steps.
    '''

    for i in range(steps):
      images, labels = self.get_torch_tensors()
      images = torch.Tensor(images)
      labels = torch.Tensor(labels)

      #if torch.cuda.is_available():
      #  images = images.float().cuda()
      #  labels = labels.float().cuda()
      #else:
      images = images.float()
      labels = labels.float()
      optimizer.zero_grad()
      
      log_ps  = self.pytorchmodel(images)
      print("log_ps shape: ", log_ps.shape)
      print("labels shape: ", labels.shape)
      loss = criterion(log_ps, labels)
      loss.backward()
      optimizer.step()



  def train(self, dataset, remaining_time_budget=None):
    steps_to_train = self.get_steps_to_train(remaining_time_budget)
    if steps_to_train <= 0:
      print_log("Not enough time remaining for training. " +
            "Estimated time for training per step: {:.2f}, "\
            .format(self.estimated_time_per_step) +
            "but remaining time budget is: {:.2f}. "\
            .format(remaining_time_budget) +
            "Skipping...")
      self.done_training = True
    else:
      msg_est = ""
      if self.estimated_time_per_step:
        msg_est = "estimated time for this: " +\
                  "{:.2f} sec.".format(steps_to_train * self.estimated_time_per_step)
      print_log("Begin training for another {} steps...{}".format(steps_to_train, msg_est))

      train_start = time.time()


      # PYTORCH
      self.training_data_iterator = self.input_function(dataset, is_training=True)
      #Training loop inside
      #criterion = nn.NLLLoss()
      criterion = nn.BCEWithLogitsLoss()  # Multilabel
      optimizer = torch.optim.Adam(self.pytorchmodel.parameters(), lr=1e-3)
      self.trainloop(criterion, optimizer, dataset, steps=steps_to_train)
      train_end = time.time()

      # Update for time budget managing
      train_duration = train_end - train_start
      self.total_train_time += train_duration
      self.cumulated_num_steps += steps_to_train
      self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
      print_log("{} steps trained. {:.2f} sec used. ".format(steps_to_train, train_duration) +\
            "Now total steps trained: {}. ".format(self.cumulated_num_steps) +\
            "Total time used for training: {:.2f} sec. ".format(self.total_train_time) +\
            "Current estimated time per step: {:.2e} sec.".format(self.estimated_time_per_step))



  def get_steps_to_train(self, remaining_time_budget):
    """Get number of steps for training according to `remaining_time_budget`.

    The strategy is:
      1. If no training is done before, train for 10 steps (ten batches);
      2. Otherwise, estimate training time per step and time needed for test,
         then compare to remaining time budget to compute a potential maximum
         number of steps (max_steps) that can be trained within time budget;
      3. Choose a number (steps_to_train) between 0 and max_steps and train for
         this many steps. Double it each time.
    """
    if not remaining_time_budget: # This is never true in the competition anyway
      remaining_time_budget = 1200 # if no time limit is given, set to 20min

    if not self.estimated_time_per_step:
      steps_to_train = 10
    else:
      if self.estimated_time_test:
        tentative_estimated_time_test = self.estimated_time_test
      else:
        tentative_estimated_time_test = 50 # conservative estimation for test
      max_steps = int((remaining_time_budget - tentative_estimated_time_test) / self.estimated_time_per_step)
      max_steps = max(max_steps, 1)
      if self.cumulated_num_tests < np.log(max_steps) / np.log(2):
        steps_to_train = int(2 ** self.cumulated_num_tests) # Double steps_to_train after each test
      else:
        steps_to_train = 0
    return steps_to_train  

  def testloop(self, dataloader):
    '''
    # PYTORCH
    testloop uses testdata to test the pytorch model and return onehot prediciton values.
    '''
    preds = []
    with torch.no_grad():
          self.pytorchmodel.eval()
          for [images] in dataloader:
            #if torch.cuda.is_available():
            #  images = images.float().cuda()
            #else:
            images = images.float()
            #print("in testloop, image shape: ", images.shape)
            log_ps = self.pytorchmodel(images)
            #ps = torch.exp(log_ps)
            sig = nn.Sigmoid()
            ps = sig(log_ps)
            #print(ps)
            #print("#"*100)
            #preds.append(top_class.cpu().numpy())
            preds.append(ps)

    preds = np.concatenate(preds)
    #onehot_preds = np.squeeze(np.eye(self.output_dim)[preds.reshape(-1)])
    return preds


  def choose_to_stop_early(self):
    """The criterion to stop further training (thus finish train/predict
    process).
    """
    # return self.cumulated_num_tests > 10 # Limit to make 10 predictions
    # return np.random.rand() < self.early_stop_proba
    batch_size = self.batch_size
    num_examples = self.metadata_.size()
    num_epochs = self.cumulated_num_steps * batch_size / num_examples
    print_log("Model already trained for {} epochs.".format(num_epochs))
    return num_epochs > self.num_epochs_we_want_to_train # Train for at least certain number of epochs then stop



  def test(self, dataset, remaining_time_budget=None):
    if self.done_training:
      return None

    if self.choose_to_stop_early():
      print_log("Oops! Choose to stop early for next call!")
      self.done_training = True
    test_begin = time.time()
    if remaining_time_budget and self.estimated_time_test and\
        self.estimated_time_test > remaining_time_budget:
      print_log("Not enough time for test. " +\
            "Estimated time for test: {:.2e}, ".format(self.estimated_time_test) +\
            "But remaining time budget is: {:.2f}. ".format(remaining_time_budget) +\
            "Stop train/predict process by returning None.")
      return None

    msg_est = ""
    if self.estimated_time_test:
      msg_est = "estimated time: {:.2e} sec.".format(self.estimated_time_test)
    print_log("Begin testing...", msg_est)

    # PYTORCH
    testloader = self.get_dataloader(dataset, batch_size=self.batch_size, train=False)
    predictions = self.testloop(testloader)

    test_end = time.time()
    # Update some variables for time management
    test_duration = test_end - test_begin
    self.total_test_time += test_duration
    self.cumulated_num_tests += 1
    self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
    print_log("[+] Successfully made one prediction. {:.2f} sec used. ".format(test_duration) +\
          "Total time used for testing: {:.2f} sec. ".format(self.total_test_time) +\
          "Current estimated time for test: {:.2e} sec.".format(self.estimated_time_test))
    return predictions



  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

#### Can contain other functions too
def print_log(*content):
  """Logging function. (could've also used `import logging`.)"""
  now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
  print("MODEL INFO: " + str(now)+ " ", end='')
  print(*content)

def get_num_entries(tensor):
  """Return number of entries for a TensorFlow tensor.
  Args:
    tensor: a tf.Tensor or tf.SparseTensor object of shape
        (batch_size, sequence_size, row_count, col_count[, num_channels])
  Returns:
    num_entries: number of entries of each example, which is equal to
        sequence_size * row_count * col_count [* num_channels]
  """
  tensor_shape = tensor.shape
  assert(len(tensor_shape) > 1)
  num_entries  = 1
  for i in tensor_shape[1:]:
    num_entries *= int(i)
  return num_entries

def crop_time_axis(tensor_4d, num_frames, begin_index=None):
  """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.
  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels]
    num_frames: An integer representing the resulted chunk (sequence) length
    begin_index: The index of the beginning of the chunk. If `None`, chosen
      randomly.
  Returns:
    A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
  """
  # pad sequence if not long enough
  pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[0], 0)
  padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

  # If not given, randomly choose the beginning index of frames
  if not begin_index:
    maxval = tf.shape(padded_tensor)[0] - num_frames + 1
    begin_index = tf.random.uniform([1],
                                    minval=0,
                                    maxval=maxval,
                                    dtype=tf.int32)
    begin_index = tf.stack([begin_index[0], 0, 0, 0], name='begin_index')

  sliced_tensor = tf.slice(padded_tensor,
                           begin=begin_index,
                           size=[num_frames, -1, -1, -1])

  return sliced_tensor

def resize_space_axes(tensor_4d, new_row_count, new_col_count):
  """Given a 4-D tensor, resize space axes to have target size.
  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels].
    new_row_count: An integer indicating the target row count.
    new_col_count: An integer indicating the target column count.
  Returns:
    A Tensor of shape [sequence_size, target_row_count, target_col_count].
  """
  resized_images = tf.image.resize_images(tensor_4d,
                                          size=(new_row_count, new_col_count))
  return resized_images

def print_log(*content):
  """Logging function. (could've also used `import logging`.)"""
  now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
  print("MODEL INFO: " + str(now)+ " ", end='')
  print(*content)