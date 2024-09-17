import struct
import numpy as np

def MNIST(ds_path: str):
  def load_mnist_images(filename):
    with open(filename, 'rb') as f:
      _, num, rows, cols = struct.unpack('>IIII', f.read(16))
      images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

  def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
      _, num = struct.unpack('>II', f.read(8))
      labels = np.fromfile(f, dtype=np.uint8)
    return labels
  
  # Load the data
  train_images = load_mnist_images(ds_path + '/train-images.idx3-ubyte')
  train_labels = load_mnist_labels(ds_path + '/train-labels.idx1-ubyte')
  test_images = load_mnist_images(ds_path + '/t10k-images.idx3-ubyte')
  test_labels = load_mnist_labels(ds_path + '/t10k-labels.idx1-ubyte')

  # Check the shapes
  print(f"train_images shape: {train_images.shape}, train_labels shape: {train_labels.shape}")
  print(f"test_images shape: {test_images.shape}, test_labels shape: {test_labels.shape}")
  
  # Preparing train, test, val splits
  val_mask = np.arange(0,60000)
  np.random.shuffle(val_mask)
  val_images = train_images[val_mask[:2000]]
  val_labels = train_labels[val_mask[:2000]]
  train_images = train_images[val_mask[2000:]]
  train_labels = train_labels[val_mask[2000:]]

  return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)