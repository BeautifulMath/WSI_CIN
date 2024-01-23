# Reference: https://github.com/binli123/dsmil-wsi 
# Download ResNet18-SSL model weight: https://github.com/ozanciga/self-supervised-histopathology

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.io import read_image
import sys, argparse, glob
import pandas as pd
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus: 
  tf.config.experimental.set_virtual_device_configuration(gpus[0], 
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5*1024)])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

class BagDataset():
    def __init__(self, image_list, transform=None):
        self.files_list = image_list
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = read_image(img)
        img = img / 255.0
        sample = {'input': img}
        if self.transform:
            sample = self.transform(sample)
        return sample 

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

def bag_dataset(args, image_list):
    transformed_dataset = BagDataset(image_list=image_list, transform=None) 
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def compute_feats(args, bags_list, i_classifier, save_path=None, SSL=True):
    i_classifier.eval()
    num_bags = len(bags_list)
    for i in range(0, num_bags):
        feats_list = []
        image_list = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.png'))
        dataloader, bag_size = bag_dataset(args, image_list)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                if SSL:
                    feats = i_classifier(patches)
                else:
                    feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy() 
                feats_list.extend(feats) 
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-1][:23]+'.csv'), index=False, float_format='%.4f')

def main():
    ## python extract_deep_features.py --backbone='resnet18'
    parser = argparse.ArgumentParser(description='Compute deep features from tiles/nucleus')
    parser.add_argument('--backbone', default='resnet18-ssl', type=str, help='Embedder backbone (resnet18, resnet18-ssl, vgg11, densenet121)')
    parser.add_argument('--image_source', default='nucleus', type=str, help='tile or nucleus')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size of dataloader')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of threads for dataloader')
    args = parser.parse_args()
    SSL = False
    if args.backbone == 'resnet18':
        feature_extractor = models.resnet18(pretrained=True)
        num_feats = 1000
        i_classifier = IClassifier(feature_extractor, num_feats, output_class=1).cuda()
    if args.backbone == 'vgg11':
        feature_extractor = models.vgg11(pretrained=True)
        num_feats = 1000
        i_classifier = IClassifier(feature_extractor, num_feats, output_class=1).cuda()
    if args.backbone == 'densenet121':
        feature_extractor = models.densenet121(pretrained=True)
        num_feats = 1000
        i_classifier = IClassifier(feature_extractor, num_feats, output_class=1).cuda()
    if args.backbone == 'resnet18-ssl':
        SSL = True
        MODEL_PATH = 'tenpercent_resnet18.ckpt'
        i_classifier = models.__dict__['resnet18'](pretrained=False)
        state = torch.load(MODEL_PATH, map_location='cuda:0')
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        i_classifier = load_model_weights(i_classifier, state_dict)
        i_classifier.fc = torch.nn.Sequential()
        i_classifier.to(device)
    if args.image_source == 'tile':
        image_folder = './tile'
    if args.image_source == 'nucleus':
        image_folder = './nucleus'

    bags_list = glob.glob(os.path.join(image_folder, '*'))
    feats_path = os.path.join('./features', args.image_source, args.backbone)
    if not os.path.exists(feats_path):
        os.makedirs(feats_path)
    compute_feats(args, bags_list, i_classifier, feats_path, SSL)

if __name__ == '__main__':
    main()