import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import ModelNet40
from model import PointNet

SEED = 13
gpus = [0]
batch_size = 32
ckp_path = './output/latest.pth'


def load_ckp(ckp_path, model):
    state = torch.load(ckp_path)
    model.load_state_dict(state['state_dict'])
    print("model load from %s" % ckp_path)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    print("Loading test dataset...")
    test_data = ModelNet40("./dataset/modelnet40_normal_resampled", train=1)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model = PointNet().to(device=device)
    if ckp_path:
        load_ckp(ckp_path, model)
        model = model.to(device)

    model.eval()

    with torch.no_grad():
        accs = []
        gt_ys = []
        pred_ys = []
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            pred_y = np.argmax(out.cpu().numpy(), axis=1)

            gt = np.argmax(y.cpu().numpy(), axis=1)
            print("pred[" + str(pred_y) + "] gt[" + str(gt) + "]")

            acc = np.sum(pred_y == gt)/ gt.size
            gt_ys = np.append(gt_ys, gt)
            pred_ys = np.append(pred_ys, pred_y)

            accs.append(acc)

        print("final acc is: " + str(np.mean(accs)))
