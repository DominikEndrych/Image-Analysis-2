from torch.utils.data import Dataset
import glob

class DatasetParking(Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform

        #images_full = [img for img in glob.glob(image_path + "/full/*.png")]     # Reading images in grayscale
        #images_free = [img for img in glob.glob(image_path + "/free/*.png")]
        #labels_full = [1] * len(images_full)
        #labels_free = [0] * len(images_free)

        #self.images = images_full + images_free
        #self.labels = labels_full + labels_free

        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


