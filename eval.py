from utils.getter import *
import argparse
import os

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config)

    metric = RetrievalScore(valset, top_k=5, retrieval_pairing='img-to-img')    

    net = get_model(args, config)

    model = Retrieval2(
            model = net, 
            device = device)

    if args.weight is not None:                
        load_checkpoint(model, args.weight)
        
    metric.update(model)
    result_dict = metric.value()
    print(result_dict)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('--max_images' , type=int, help='max number of images', default=10000)
    parser.add_argument('--weight' , type=str, default=None, help='project file that contains parameters')
    parser.add_argument('--freeze_cnn', action='store_true', help='whether to freeze the backbone')
    args = parser.parse_args()
    config = Config(os.path.join('configs','config2.yaml'))

    train(args, config)
    


