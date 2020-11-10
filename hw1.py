import torch
from torch import nn
import torchvision.models as models
import loader
from torch.utils.data import DataLoader
import argparse
import torch.optim as optim
import csv
import os
from torchvision import transforms as trns
import PIL.Image as Image
def create_nn():
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Linear(in_features=2048, out_features=196)
    nn.init.kaiming_normal_(resnet.fc.weight, mode='fan_in')
    return resnet

# train model
def train(args, model, epoch, optimizer, trainloader, device, criterion=nn.CrossEntropyLoss()):
    total_loss = 0.0
    total_size = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        if args.gpu:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), total_loss / total_size, loss))

            with open('loss.csv', 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch+batch_idx/len(trainloader), total_loss / total_size, loss])
    torch.save(model.state_dict(), "./checkpoints/model_{}".format(str(epoch)))



# test model
def test(model):
    test_transform = trns.Compose([
        trns.Resize((512, 512)),
        trns.CenterCrop((448, 448)),
        trns.ToTensor(),
    	trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = "./testing_data/testing_data/"
    with open("test.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id","label"])
    for filename in os.listdir(test_dir):
        img_pil = Image.open(os.path.join(test_dir, filename)).convert('RGB')
        img_tnsr = test_transform(img_pil)
        img_tnsr = img_tnsr.unsqueeze(0)
        output = model(img_tnsr)
        output = output[0].cpu()
        output = output.tolist()
        idx = output.index(max(output))
        label_id = []
        with open('label_dict.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                label_id.append(row[0])

        with open("test.csv", "a+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename[0:6], label_id[idx]])




# parse arguments
def get_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--gpu", action="store_true", default=True)
    arg_parser.add_argument("--lr", default=0.01)
    arg_parser.add_argument("--mode", default="train")
    arg_parser.add_argument("--log_interval", default=10)
    arg_parser.add_argument("--epoch", default=100)
    arg_parser.add_argument("--model", default="checkpoints/model_99")
    args = arg_parser.parse_args()
    return args


def main():
    args = get_arguments()
    device = torch.device("cuda:0")
    net = create_nn()

    # set gpu id
    net = nn.DataParallel(net, device_ids=[0, 1, 2])
    if args.gpu:
        net.to(device)
    
    if args.mode == "train":
        net.load_state_dict(torch.load("./checkpoints/model_149"))
        train_data = loader.trainset()
        trainloader = DataLoader(train_data, batch_size=50, shuffle=True)

        # optimizer = optim.Adam(net.parameters(), lr=args.lr)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        for epoch in range(1, args.epoch+1):
            scheduler.step(epoch)
            train(args, net, epoch, optimizer, trainloader, device)

    elif args.mode == "test":
        net.load_state_dict(torch.load(args.model))
        net.eval()
        test(net)


if __name__ == "__main__":
    main()