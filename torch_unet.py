# ------------------------------------------------------------------------------
# Name:        Image Segmentation with U-Net bached on pytorch
# Purpose:	   Strait forward way to set directories to a unet train and evaluate it all with setting appropriate parametters..
#              This code is for the Carvana dataset challenge to make mask for the car Images.
#
# Author:      Morteza Heidari
#
# Created:     01/08/2022
# ------------------------------------------------------------------------------
# used padded convolutions to build a unet for image segmentation
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
# import
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import shutil

#  ##### ***** Haperparameters ***** #############################################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHES = 10
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
AUGMENT = True  # apply augmentation to the images
TRAIN_IMG_DIR = './carvana-image-masking-challenge/train/train'
TRAIN_MASK_DIR = './carvana-image-masking-challenge/train_masks/train_masks'
TEST_IMG_DIR = './carvana-image-masking-challenge/test/test'
TEST_MASK_DIR = './carvana-image-masking-challenge/test_masks/test_masks'
MODEL_DIR = './models'
PREDICTED_IMG_DIR = './predicted_img'

###############################################################################
#  ####****** UTILITY FUNCTIONS ***** #########################################
###############################################################################


class utility_functions():
    def load_checkpoint(self, checkpoint, model):

        model.load_state_dict(checkpoint['state_dict'])
        print(" Checkpoints are loaded!")
        return model

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')
            print("Best model is saved!")

    def dice_coeff(self, pred, target):
        smooth = 1e-9
        intersection = (pred * target).sum()
        score = (2. * intersection) / (pred.sum() + target.sum() + smooth)
        return score

    def accuracy_checker(self, loader, model):
        # *******************************************************
        model.eval()
        correct = 0
        total = 0
        pixels = 0
        dice = 0
        with torch.no_grad():
            for data in tqdm(loader):
                images, labels = data
                images = images.to(DEVICE)
                # labels = labels.to(DEVICE).squeeze(1)
                labels = labels.to(DEVICE).unsqueeze(1)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #
                predicts = torch.sigmoid(outputs)
                predicts = (predicts > 0.5).float()
                # score += (predicts == labels).sum().item() # line 63
                pixels += torch.numel(predicts)  # numel is number of elements in tensor
                dice += self.dice_coeff(predicts, labels)
        model.train()
        print("Accuracy of the network on the test images: %d %% " % (100 * correct / pixels))
        print("Dice score of the network on the test images: %d %% " % (100 * dice / len(loader)))
        print("Pixels of the network on the test images: %d " % (pixels))
        # return correct / total, dice / total
        # *******************************************************
        # num_correct = 0
        # num_pixels = 0
        # dice_score = 0
        # model.eval()

        # with torch.no_grad():
        #     for x, y in loader:
        #         x = x.to(DEVICE)
        #         y = y.to(DEVICE).unsqueeze(1)
        #         preds = torch.sigmoid(model(x))
        #         preds = (preds > 0.5).float()
        #         num_correct += (preds == y).sum()
        #         num_pixels += torch.numel(preds)
        #         dice_score += (2 * (preds * y).sum()) / (
        #             (preds + y).sum() + 1e-8
        #         )

        # print(
        #     f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        # )
        # print(f"Dice score: {dice_score/len(loader)}")
        # model.train()

    def get_loaders(self, train_dir,
                    train_mask_dir,
                    test_dir,
                    test_mask_dir,
                    batch_size,
                    train_transform, val_transforms,
                    num_workers=NUM_WORKERS,
                    pin_memory=True):
        train_dataset = Mydata(train_dir, train_mask_dir, train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_dataset = Mydata(test_dir, test_mask_dir, val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader

    def save_predicted_img(self, loader, model, save_dir="output_images/"):
        model.eval()
        # for data in tqdm(loader):
        #     images, labels = data
        #     images = images.to(DEVICE)
        #     labels = labels.to(DEVICE).squeeze(1)
        #     outputs = model(images)
        #     predicts = torch.sigmoid(outputs)
        #     predicts = (predicts > 0.5).float()
        #     predicts = predicts.cpu().numpy()
        #     for i in range(len(predicts)):
        #         img = Image.fromarray(predicts[i] * 255)
        #         img.save(os.path.join(save_dir, 'predicted_img_' + str(i) + '.png'))
        # model.train()
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=DEVICE)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{save_dir}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{save_dir}{idx}.png")

        model.train()


###############################################################################
#  #### ***** Data Loader and Data Augmentation ***** #########################
###############################################################################


class Mydata(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)
        self.img_list.sort()
        self.mask_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_path, self.img_list[idx]))
        mask = Image.open(os.path.join(self.mask_path, self.mask_list[idx]))
        image = image.convert('RGB')
        mask = mask.convert('L')
        image = np.array(image)
        mask = np.array(mask)
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        mask[mask == 255.0] = 1
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        return image, mask


###############################################################################
#  #### ***** UNET ***** ######################################################
###############################################################################


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):  # in_ch = 3, out_ch = 1 is conisedred for a 3 channele input and binary segmentation
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        self.downs.append(DoubleConv(in_ch, 64))
        self.downs.append(DoubleConv(64, 128))
        self.downs.append(DoubleConv(128, 256))
        self.downs.append(DoubleConv(256, 512))
        # self.downs.append(DoubleConv(512, 1024))
        #
        self.ups.append(nn.ConvTranspose2d(1024, 512, 2, 2))
        self.ups.append(DoubleConv(1024, 512))
        self.ups.append(nn.ConvTranspose2d(512, 256, 2, 2))
        self.ups.append(DoubleConv(512, 256))
        self.ups.append(nn.ConvTranspose2d(256, 128, 2, 2))
        self.ups.append(DoubleConv(256, 128))
        self.ups.append(nn.ConvTranspose2d(128, 64, 2, 2))
        self.ups.append(DoubleConv(128, 64))
        self.bottleneck = DoubleConv(512, 1024)
        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

        # self.conv1 = DoubleConv(in_ch, 64)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = DoubleConv(64, 128)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = DoubleConv(128, 256)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv4 = DoubleConv(256, 512)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv5 = DoubleConv(512, 1024)
        # self.up6 = nn.ConvTranspose2d(1024, 512, kernek_size=2, stride=2)
        # self.conv6 = DoubleConv(1024, 512)
        # self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.conv7 = DoubleConv(512, 256)
        # self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.conv8 = DoubleConv(256, 128)
        # self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.conv9 = DoubleConv(128, 64)
        # self.bottleneck = DoubleConv(512, 1024)
        # self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        skip_connection = []
        for i in range(len(self.downs)):
            x = self.downs[i](x)
            skip_connection.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connection = skip_connection[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_conn = skip_connection[i // 2]
            if x.shape != skip_conn.shape:
                x = TF.resize(x, skip_conn.shape[2:])  # resize x to match skip_conn based on height and width
            concat = torch.cat([x, skip_conn], dim=1)
            # concat = torch.cat((skip_conn, x), dim = 1)
            x = self.ups[i + 1](concat)
        x = self.out(x)
        return x


###############################################################################
#  #### ***** TRAININ & VALIDATION***** #######################################
###############################################################################
def train_model(model, dataloader, optimizer, scaler, loss_fun, num_epochs=25):
    loop = tqdm(dataloader)
    for batch_idx, (img, mask) in enumerate(loop):
        img = img.to(device=DEVICE, dtype=torch.float)
        # mask = mask.to(device=DEVICE, dtype=torch.float, unsqueeze=True)
        mask = mask.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            output = model(img)
            loss = loss_fun(output, mask)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_description(f'loss: {loss.item():.4f}')
        loop.set_postfix(loss=loss.item())
        # print(f'loss: {loss.item():.4f}')


def test():
    model = UNET(in_ch=1, out_ch=1)
    x = torch.randn((3, 1, 512, 512))  # batch_size, channel, height, width
    y = model(x)
    print(y.shape)
    print(x.shape)
    assert y.shape == x.shape


def main():
    if AUGMENT:
        train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
    else:
        train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    model = UNET(in_ch=3, out_ch=1)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fun = nn.BCEWithLogitsLoss()  # if it is a 3 channel output (for multiple classes) use a cross entropy loss
    util = utility_functions()
    train_dataloader, validations_dataloader = util.get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        TEST_IMG_DIR, TEST_MASK_DIR,
        BATCH_SIZE, train_transform, val_transforms,
        NUM_WORKERS, PIN_MEMORY
    )

    if LOAD_MODEL:
        util.load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    util.accuracy_checker(validations_dataloader, model)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHES):
        # train model
        train_model(model, train_dataloader, optimizer, scaler, loss_fun)
        # save model
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        util.save_checkpoint(checkpoint, is_best=False)
        # check accuracy
        util.accuracy_checker(validations_dataloader, model)

        torch.save(model.state_dict(), f"{MODEL_DIR}/{epoch}.pt")
        print(f"Saved model to {MODEL_DIR}/{epoch}.pt")
        # accuracy evaluation
        util.save_predicted_img(validations_dataloader, model, save_dir=PREDICTED_IMG_DIR)


if __name__ == '__main__':
    main()
