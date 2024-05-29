import torch
import os
import regex as re
import matplotlib.pyplot as plt

def get_index(dir):
    files = os.listdir(dir)
    if len(files) == 0:
        return 0
    indexes = []
    for file in files:
        index = re.findall(r'\d+', file)
        if index:
            indexes.append(int(index[0]))
    if len(indexes) == 0:
        return 0
    return max(indexes)+1

def plot_results(train_loss, dev_loss, metrics, dir="SA/plots/"):
    # test time values
    precision, recall, f1_score = metrics[-1]
    plt.plot(train_loss, label='train_loss')
    plt.plot(dev_loss, label='dev_loss')
    plt.legend()
    idx = get_index(dir)
    with open(dir + f'metrics_{idx}.txt', 'w') as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1_score}\n")
    
    plt.savefig(dir + f'loss_{idx}.png')

def save_model(model, optimizer, dir='SA/bin'):
    index = get_index(dir)
    torch.save(model.state_dict(), f"{dir}/model_{index}.pt")
    torch.save(optimizer.state_dict(), f"{dir}/optimizer_{index}.pt")
    print(f"Model saved at {dir}/model_{index}.pt")
    print(f"Optimizer saved at {dir}/optimizer_{index}.pt")
    with open(f"{dir}/optim{get_index(dir)}.txt", 'w') as f:
        f.write(optimizer.state_dict().__str__() + '\n')
