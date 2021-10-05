from utils.getter import *
import argparse

parser = argparse.ArgumentParser('Training Object Detection')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
parser.add_argument('--top_k', type=int, default=10, help='Retrieve top k results')
parser.add_argument('--weight', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.globals['gpu_devices']
    num_gpus = len(config.globals['gpu_devices'].split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    devices_info = get_devices_info(config.globals['gpu_devices'])
    
    valloader = get_instance(config.valloader)
    valset = valloader.dataset

    net = get_instance(config.model, device=device)

    metric = RetrievalScore(
            valset, valset, 
            max_distance = 1.3,
            top_k=args.top_k,
            dimension=config.model['args']['d_embed'],
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
    config = get_config(args.weight)
    if config is None:
        config = Config(os.path.join('configs',f'{args.config}'))
    else:
        print("Load configs from weight")   

    train(args, config)