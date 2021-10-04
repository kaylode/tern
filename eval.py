from utils.getter import *
import argparse

parser = argparse.ArgumentParser('Training Object Detection')
parser.add_argument('--top_k', type=int, default=10, help='Retrieve top k results')
parser.add_argument('--weight', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    devices_info = get_devices_info(config.gpu_devices)
    
    _, valset, _, _ = get_dataset_and_dataloader(config)

    net = get_cross_modal(config.model)

    metric = RetrievalScore(
            valset, valset, 
            max_distance = 1.3,
            top_k=10,
            dimension=config.model['d_embed'],
            save_results=True)

    model = Retriever(model = net, device=device)
  
    if args.weight is not None:                
        load_checkpoint(model, args.weight)
        
    print()
    print("##########   DATASET INFO   ##########")
    print("Valset: ")
    print(valset)
    print()
    print(config)
    print(f'Evaluating with {num_gpus} gpu(s): ')
    print(devices_info)

    metric.update(model)
    metric_results = metric.value()
    print(metric_results)

if __name__ == '__main__':
    
    args = parser.parse_args()
    config = Config(os.path.join('configs','config.yaml'))

    train(args, config)