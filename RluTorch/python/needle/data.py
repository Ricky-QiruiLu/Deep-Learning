import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, 1)
        return  img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        result = np.zeros(img.shape)
        
        shift_x = -shift_x
        shift_y = -shift_y
        if shift_x > 0:
            if shift_y > 0:
                result[shift_x:, shift_y:, :] = img[:-shift_x, :-shift_y, :]
            elif shift_y < 0:
                result[shift_x:, :shift_y, :] = img[:-shift_x, -shift_y:, :]
            else:
                result[shift_x:, :, :] = img[:-shift_x, :, :]
        
        elif shift_x < 0:
            if shift_y > 0:
                result[:shift_x, shift_y:, :] = img[-shift_x:, :-shift_y, :]
            elif shift_y < 0:
                result[:shift_x, :shift_y, :] = img[-shift_x:, -shift_y:, :]
            else:
                result[:shift_x, :, :] = img[-shift_x:, :, :]
        else:
            if shift_y > 0:
                result[:, shift_y:, :] = img[:, :-shift_y, :]
            elif shift_y < 0:
                result[:, :shift_y, :] = img[:, -shift_y:, :]
            else:
                result[:, :, :] = img[:, :, :]
        
        return result


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
                                          range(self.batch_size, len(self.dataset), self.batch_size))
        
        self.batch_idx = 0
        return self

    def __next__(self):
        try:
            
            curr_batch = self.ordering[self.batch_idx]
            curr_sample = [self.dataset[i] for i in curr_batch]
            
            result = []
            for i in range(len(curr_sample[0])):
                temp_tensor = Tensor(np.stack([sample[i] for sample in curr_sample]), dtype=np.float32)
                result.append(temp_tensor)
            
            self.batch_idx += 1
            return tuple(result)
        
        except:
            raise StopIteration
        

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms
        
        self.x, self.y = parse_mnist(self.image_filename, self.label_filename)

    def __getitem__(self, index) -> object:
        raw_item = self.x[index]
        raw_item = self.apply_transforms(raw_item)
        return raw_item, self.y[index]
       
    def __len__(self) -> int:
        return len(self.y)
       
class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])

# helper for mnist data loader
def parse_mnist(image_filesname, label_filename):
    import gzip
    import struct
    
    size_lf_mega = 8
    size_if_mega = 16
    
    size_pixel = 784
    
    labels = []
    pixels = []
    
    with gzip.open(image_filesname, 'rb') as image_f:
      if_mega_data_b = image_f.read(size_if_mega)
      
      while True:
        block = image_f.read(size_pixel)   
        if not block:
            break
    
        temp_pixel = list(struct.unpack('B'*size_pixel, block))
        pixels.append(temp_pixel)
    
    X = np.array(pixels, dtype=np.float32)
    
    for i in range(X.shape[1]):
      if X[:, i].max() != 0:
        X[:, i] = X[:, i] / 255     # nomralization
        
    X = X.reshape((-1, 28, 28, 1))  # -1 represent total dim / 28 / 28
    
    # label loader
    with gzip.open(label_filename, 'rb') as label_f:
      lf_mega_data_b = label_f.read(size_lf_mega)
    #   print(lf_mega_data_b)
    
      while True:
        block = label_f.read(1)   
        if not block:
            break
    
        temp_label = struct.unpack('B', block)[0]
        labels.append(temp_label)
    
    
    y = np.array(labels, dtype=np.uint8)
    return (X, y)
        

# helper for CIFAR10Dataset
def parse_CIFAR10(file_name):
  """
  reference by https://www.cs.toronto.edu/~kriz/cifar.html
  """
  def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
  
  data = unpickle(file_name)

  X_raw = data[b'data']
  Y_raw = data[b'labels']
  X_shape = (3, 32, 32)

  X = []
  Y = []
  for temp_x, temp_y in zip(X_raw, Y_raw):
    X.append(temp_x.reshape(X_shape))
    Y.append(temp_y)
  
  X = np.array(X, dtype=np.float32)
  Y = np.array(Y, dtype=np.int8)
  return X, Y

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        self.base_folder = base_folder
        self.train = train

        self.X = np.array([])
        self.y = np.array([])

        # training mode
        if self.train:
          train_files = ["/data_batch_1", "/data_batch_2", "/data_batch_3", "/data_batch_4", "/data_batch_5"]
        else:
          train_files = ["/test_batch"]

        for file in train_files:
          curr_train_file = self.base_folder + file
          curr_x, curr_y = parse_CIFAR10(curr_train_file)

          if len(self.X) == 0:
            self.X = curr_x
            self.y = curr_y
          else:
            self.X = np.concatenate((self.X, curr_x), axis=0)
            self.y = np.concatenate((self.y, curr_y), axis=0)
        
    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        return self.X[index], self.y[index]
        
    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return len(self.y)
        

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])



class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        if word not in self.word2idx:
          idx = len(self.idx2word)

          self.word2idx[word] = idx
          self.idx2word.append(word)
        else:
          idx = self.word2idx[word]

        return idx
        
    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return len(self.word2idx)
        


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ids = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break

                words = line.split() + ['<eos>']

                for word in words:
                    ids.append(self.dictionary.add_word(word))

        return ids
        

def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    n_rows = len(data) // batch_size

    result = []

    for i in range(batch_size):
      result.append(data[i * n_rows: i * n_rows + n_rows])

    result = np.array(result, dtype=dtype).reshape(batch_size, -1).T

    return result
    

def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    seq_len = min(bptt, len(batches) - 1 - i)
    data = batches[i: i + seq_len]

    target = batches[i + 1: i + seq_len + 1].reshape((data[0].shape[0] * seq_len,))
    return Tensor(data, device=device, dtype=dtype), Tensor(target, device=device, dtype=dtype)
    

