import os
import PIL.Image as Image
import numpy as np

class Dataset:

    def __init__(self, datapath):
        self.datapath = datapath

        with open(os.path.join(datapath, 'celeba_filtered.txt'), 'r') as f:
            id_filename_attribute = f.readlines()
        self.attributeNames = id_filename_attribute[0].split()[2:]
        ids, filenames, attributes = [], [], []
        for item in id_filename_attribute[1:]:
            item_split = item.split()
            ids.append(item_split[0])
            filenames.append(item_split[1])
            attributes.append([int(attr) for attr in item_split[2:]])
        self.ids = ids
        self.filenames = filenames
        self.attributes = attributes
        
        with open(os.path.join(datapath, 'partition.txt'), 'r') as f:
            partitions = f.readlines()
        train_indices, test_indices = [], []
        for i, item in enumerate(partitions):
            item_split = item.split()
            assert item_split[0] == self.filenames[i]
            if item_split[1] == '0':
                train_indices.append(i)
            elif item_split[1] == '1':
                test_indices.append(i)
            else:
                raise ValueError
        self.train_indices = train_indices
        self.test_indices = test_indices

    def get_imgs(self, indices):
        imgs = []
        for index in indices:
            img = Image.open(os.path.join(
                self.datapath, 'celeba_filtered', self.filenames[index]))
            img = np.asarray(img)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        imgs = imgs.astype(np.float32) / 128. - 1.
        N, H, W, C = imgs.shape
        return imgs.reshape(N, -1)
    
    def get_attrs(self, indices):
        attrs = []
        for index in indices:
            attr = self.attributes[index]
            attrs.append(attr)
        attrs = np.stack(attrs, axis=0)
        return attrs
    
    def get_labels(self, indices):
        labels = []
        label_mapping = {'3029':0, '8657':1, '1341':2, '8386':3, '6364':4, '3376':5, '7541':6, '4987':7, '3047':8, '2496':9}
        for index in indices:
            label = label_mapping[self.ids[index]]
            labels.append(label)
        labels = np.stack(labels, axis=0)
        return labels

    def get_train_data(self):
        X_train = self.get_attrs(self.train_indices)
        Y_train = self.get_labels(self.train_indices)
        return X_train, Y_train
    
    def get_test_data(self):
        X_test = self.get_attrs(self.test_indices)
        Y_test = self.get_labels(self.test_indices)
        return X_test, Y_test
    
if __name__ == "__main__":

    ## test dataset
    data = Dataset("./celeba")
    train_imgs, train_labels = data.get_train_data()
    test_imgs, test_labels = data.get_test_data()
    breakpoint()