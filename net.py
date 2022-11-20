import torch
from torch import nn
import cv2
import numpy as np
import os
import torchvision.transforms.functional as TF
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import glob
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.fft as fft
import matplotlib.cm as mpl_color_map
import torch.nn.functional as F

NUM_EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NOISE_FACTOR = 0.5
s=256

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __getitem__(self, index):
        path = self.image_paths[index]
        gt = cv2.imread(path.replace('.npy', '.png'),0)
        #gt = cv2.normalize(gt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        path_transformed = os.path.join(os.path.dirname(os.path.dirname(path)),"images", os.path.basename(path))#.replace(".png",".npy"))
        image_transformed = np.array(cv2.imread(path_transformed,0), dtype=np.float32)
        #image_transformed = cv2.normalize(image_transformed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        gt = cv2.resize(gt,dsize=(s,s)).astype(np.float32)
        image_transformed = cv2.resize(image_transformed,dsize=(s,s)).astype(np.float32)
        gt /= np.max(gt)
        image_transformed /= np.max(image_transformed)
        gt_fft = np.fft.fftshift(np.fft.fft2(gt), axes=(0, 1))
        image_transformed_fft = np.fft.fftshift(np.fft.fft2(image_transformed), axes=(0, 1))
        # transformations, e.g. Random Crop etc.
        # Make sure to perform the same transformations on image and target
        # Here is a small example: https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7?u=ptrblck
        #image_transformed = cv2.GaussianBlur(gt,(9,9),0)
        x, y = TF.to_tensor(gt_fft), TF.to_tensor(image_transformed_fft)
        return x, y

    def __len__(self):
        return len(self.image_paths)

class TrainableEltwiseLayer(nn.Module):
  def __init__(self, n, h, w):
    super(TrainableEltwiseLayer, self).__init__()
    self.weights = nn.Parameter(torch.Tensor(1, n, h, w))  # define the trainable parameter
    self.weights2 = nn.Parameter(torch.Tensor(n, n, h, w))  # define the trainable parameter

  def forward(self, x):
    # assuming x is of size b-1-h-w
    x = x * self.weights  #
    x = F.relu(x)
    x = x * self.weights2  #
    return x
def save_decoded_image(img, name):
    img = img.view(img.size(0), 4, s, s)
    save_image(img, name)
def save_decoded_heatmap(img, name):
    #img = img.view(img.size(0), 4, s, s)
    save_image(img, name)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)
if __name__ == '__main__':
    w_image =1.0
    w_fft = 1.0
    w_tv = 1e-5
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # the optimizer
    model = TrainableEltwiseLayer(n=1, h=256, w=256)  # instantiate a model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weights.data)  # init for conv layers
        if isinstance(m, TrainableEltwiseLayer):
            nn.init.constant_(m.weights.data, 1)  # init for eltwise layers
            # nn.init.normal_(m.weights.data, mean=1, std=0.1)  # init for eltwise layers

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    dir_path_train = r"trainset/gt"
    paths_train = glob.glob(os.path.join(dir_path_train, "*.png"))

    train_set = MyDataset(paths_train)

    dir_path_test = r"testset/gt"
    paths_test = glob.glob(os.path.join(dir_path_test, "*.png"))
    test_set = MyDataset(paths_test)
    #train_set, test_set = torch.utils.data.random_split(dataset, [120, 23])
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            model.train()
            img, img_noisy = data

            optimizer.zero_grad()
            outputs = model(img_noisy)
            #loss = criterion(torch.abs(fft.ifft2(outputs)), torch.abs(fft.ifft2(img)))
            loss = criterion(torch.abs(outputs), torch.abs(img))
            #loss = criterion(torch.real(outputs), torch.real(img))
            #loss = w_fft*criterion(torch.abs(outputs), torch.abs(img)) + w_image*criterion(torch.abs(fft.ifft2(outputs)),
            #                                                                 torch.abs(fft.ifft2(img))) + total_variation_loss(outputs, w_tv)

            # backpropagation
            loss.backward()
            # update the parameters
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch + 1, NUM_EPOCHS, loss))

        my_lr_scheduler.step()

        if epoch % 10 == 0:


            test_loss = 0.0
            for data in testloader:
                model.eval()
                img, img_noisy = data
                outputs = model(img_noisy)
                #loss = criterion(torch.abs(fft.ifft2(outputs)), torch.abs(fft.ifft2(img)))
                loss = criterion(torch.abs(outputs), torch.abs(img))
                #loss = criterion(torch.real(outputs), torch.real(img))
                #loss = w_fft * criterion(torch.abs(outputs), torch.abs(img)) + w_image * criterion(
                #    torch.abs(fft.ifft2(outputs)),
                #    torch.abs(fft.ifft2(img))) + total_variation_loss(outputs, w_tv)
                test_loss += loss.item()
            # plt.subplot(1, 3, 1)
            # plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(img[0, 0, :, :].detach().numpy()))))
            # plt.title("GT")
            # plt.axis('off')
            # plt.subplot(1, 3, 2)
            # plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(img_noisy[0, 0, :, :].detach().numpy()))))
            # plt.title("Input")
            # plt.axis('off')
            # plt.subplot(1, 3, 3)
            # plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(outputs[0, 0, :, :].detach().numpy()))))
            # plt.title("Output")
            # plt.axis('off')
            #plt.show()
            print('-' * 50)
            print('Epoch {} of {}, Test Loss: {:.6f}'.format(
                epoch + 1, NUM_EPOCHS, test_loss / len(testloader)))
            print('-' * 50)
            cm = plt.get_cmap('hot')

            # Apply the colormap like a function to any array:
            img_noisy_heatmap = cm(torch.abs(fft.ifft2(img_noisy)).cpu().data)
            img_noisy_heatmap = torch.tensor(np.squeeze(np.transpose(img_noisy_heatmap,axes=[0,4,2,3,1])))
            outputs_heatmap = cm(torch.abs(fft.ifft2(outputs)).cpu().data)
            outputs_heatmap = torch.tensor(np.squeeze(np.transpose(outputs_heatmap,axes=[0,4,2,3,1])))
            cm = plt.get_cmap('viridis')
            weights_heatmap = cm(torch.log(torch.abs(model.weights.cpu().data) + 1e-12)[0, 0])
            plt.imsave('./Saved_Images2/weights{}.png'.format(epoch),weights_heatmap)
            #weights_heatmap = torch.tensor(weights_heatmap)
            save_decoded_image(img_noisy_heatmap, name='./Saved_Images2/noisy{}.png'.format(epoch))
            save_decoded_image(outputs_heatmap, name='./Saved_Images2/denoised{}.png'.format(epoch))
            #save_decoded_image(weights_heatmap, name='./Saved_Images2/weights{}.png'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, f'./weights/e{epoch}.pth')


