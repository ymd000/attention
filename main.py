''' 必要なモジュールのインポート '''
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch.nn.functional as F
from plot import BagImageVisualizer, TrainingHistoryPlotter, AttentionVisualizer

''' bagの作成 '''
class MNISTBagDataset():
    def __init__(self, dataset, num_images_per_bag=16):
        """
        dataset: PyTorch Dataset (MNIST)
        num_images_per_bag: 各バッグに含まれる画像数
        """
        self.dataset = dataset
        self.num_images_per_bag = num_images_per_bag

    def __len__(self):
        # バッグの数
        return len(self.dataset) // self.num_images_per_bag

    def __getitem__(self, idx):
        # 1つのバッグを取り出すインデックス
        start_idx = idx * self.num_images_per_bag
        end_idx = start_idx + self.num_images_per_bag
        
        # 16枚の画像を取得して1つのバッグとしてまとめる
        images = []
        labels = []
        for i in range(start_idx, end_idx):
            image, label = self.dataset[i]
            images.append(image)
            labels.append(label)
        
        # 16枚の画像を1つのテンソルにまとめる (16, 1, 28, 28)
        images = torch.stack(images)
        labels = torch.tensor(labels)  # ラベルも同様にまとめる

        # ラベルに0が含まれているかどうかで新しいラベルを作成
        if 0 in labels:
            bag_label = torch.tensor(1)  # 0が含まれていれば 1
        else:
            bag_label = torch.tensor(0)  # 含まれていなければ 0

        return images, bag_label


''' バッグとしてデータセットを作成 '''
class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, data_dir='./data', num_images_per_bag=16):
        super().__init__()

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_images_per_bag = num_images_per_bag
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def prepare_data(self):
        # データセットをダウンロードする場合など
        self.train_val_dataset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=self.transform)
    
    def setup(self, stage=None):
        # データセットをセットアップ（分割など）
        self.train_val_bagged_dataset = MNISTBagDataset(self.train_val_dataset, num_images_per_bag=self.num_images_per_bag)
        self.test_bagged_dataset = MNISTBagDataset(self.test_dataset, num_images_per_bag=self.num_images_per_bag)

        train_size = int(0.7 * len(self.train_val_bagged_dataset))
        val_size = len(self.train_val_bagged_dataset) - train_size
        self.train_bagged_dataset, self.val_bagged_dataset = random_split(self.train_val_bagged_dataset, [train_size, val_size])
    
    def train_dataloader(self):
        # トレーニングデータ用のデータローダー
        self.train_bagged_dataloader = DataLoader(self.train_bagged_dataset, batch_size=self.batch_size, shuffle=True)
        return DataLoader(self.train_bagged_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        # バリデーションデータ用のデータローダー
        self.val_bagged_dataloader = DataLoader(self.val_bagged_dataset, batch_size=self.batch_size)
        return DataLoader(self.val_bagged_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        # テストデータ用のデータローダー
        self.test_bagged_dataloader = DataLoader(self.test_bagged_dataset, batch_size=self.batch_size)
        return DataLoader(self.test_bagged_dataset, batch_size=self.batch_size)
    
    def print_shape(self):
        dataloader = self.train_bagged_dataloader
        print("トレーニングデータセットの長さ（バッチ数）", len(dataloader))
        # print(dataset)
        images, bag_label = next(iter(dataloader))
        print("トレーニングデータのバッグのshape:", images.shape)  # 期待される形: (64, 16, 1, 28, 28)
        # print("トレーニングデータの最初のバッチのラベルのshape", bag_label.shape)  # 期待される形: ()
        print("最初のバッグのラベル:", bag_label[0])  # 期待される値: 0 or 1

        dataloader = self.val_bagged_dataloader
        print("バリデーションデータセットの長さ（バッチ数）", len(dataloader))
        images, bag_label = next(iter(dataloader))
        print("バリデーションデータの最初のバッチのshape:", images.shape)  # 期待される形: (16, 1, 28, 28)
        # print("バリデーションデータの最初のバッチのラベルのshape", bag_label.shape)  # 期待される形: ()
        print("最初のバッグのラベル:", bag_label[0])  # 期待される値: 0 or 1

        dataloader = self.test_bagged_dataloader
        print("テストデータセットの長さ（バッチ数）", len(dataloader))
        images, bag_label = next(iter(dataloader))
        print("テストデータの最初のバッチのshape:", images.shape)  # 期待される形: (バッチサイズ, 16, 1, 28, 28)
        # print("テストデータの最初のバッチのラベルのshape", bag_label.shape)  # 期待される形: ()
        print("最初のバッグのラベル:", bag_label[0])  # 期待される値: 0 or 1

    def set(self):
        """
        一連の関数をまとめて実行するメソッド
        """
        print("Preparing data...")
        self.prepare_data()

        print("Setting up data...")
        self.setup()

        print("Getting train data loader...")
        train_loader = self.train_dataloader()
        print(f"Train batch size: {len(train_loader.dataset)}")

        print("Getting validation data loader...")
        val_loader = self.val_dataloader()
        print(f"Validation batch size: {len(val_loader.dataset)}")

        print("Getting test data loader...")
        test_loader = self.test_dataloader()
        print(f"Test batch size: {len(test_loader.dataset)}")

        # print("Printing dataset shapes...")
        # self.print_shape()


''' Attention Layer based '''
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        """
        w : attention vector パッチごとの重みを示すベクトル shape:[attention_dim] 初期値は平均０分散１の正規分布に従う
        V : 元の画像のバッグを重みに変換するための行列（２次元目は画像特徴量を平坦化したもの）
        """
        self.w = nn.Parameter(torch.randn(attention_dim))  # Attention vector
        self.V = nn.Parameter(torch.randn(attention_dim, input_dim))  # Projection matrix 

    def forward(self, h):
        """
        h : バッグ内の画像の特徴量を示す行列 shape: [batch_size, num_images, input_dim]
        """
        # Project the embeddings to attention space
        matrix = self.V
        self.V_trandposed = matrix.T
        projections = torch.matmul(h, self.V_trandposed)  # Shape: [batch_size, num_images, attention_dim] V の転置行列との積をとった
        
        # Apply tanh non-linearity
        projections = torch.tanh(projections)  # Shape: [batch_size, num_images, attention_dim]
        
        # Compute attention scores (shape [batch_size, num_images])
        attention_scores = torch.matmul(projections, self.w)  # Shape: [batch_size, num_images]
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: [batch_size, num_images]
        
        # Aggregate the embeddings using the attention weights
        attended_values = torch.sum(attention_weights.unsqueeze(-1) * h, dim=1)  # Shape: [batch_size, input_dim] 
        """
        attention_weights に1次元加えたものと x の積を２次元に基づいて足し合わせたもの （＝ アダマール積））
        """
        
        return attended_values, attention_weights


''' Bagged Attention Model (with PyTorch Lightning) '''
class BaggedAttentionModel(pl.LightningModule):
    def __init__(self, num_images_per_bag=16, input_channels=1, output_channels=64, attention_dim=64, num_classes=1):
        super(BaggedAttentionModel, self).__init__()
        
        # CNN layers for each image
        self.cnn = nn.Sequential(
        #     nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1), # in_channels, out_channels, kernel_size, padding
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)  # 2*2で最大値を取ることで14*14の画像になる
        # )
        
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # Conv1: 32 filters, 3x3 kernel
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Max Pooling (2x2)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv2: 64 filters, 3x3 kernel
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Max Pooling (2x2)
            
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),  # Conv3: 64 filters, 3x3 kernel
            nn.ReLU(),
        )

        # number of images in a bag
        self.num_images_per_bag = num_images_per_bag

        # Attention layer
        self.input_dim = output_channels * 7 * 7  # 画像の１ピクセルごとに32の情報量がある
        self.attention = AttentionLayer(self.input_dim, attention_dim)  # 画像の特徴量を平坦化したものを処理するため、32 * 7 * 7

        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(self.input_dim, 128) # input_dimから128へ線形変換
        self.fc2 = nn.Linear(128, num_classes)

        # initialize list of outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        batch_size, num_images, channels, height, width = x.size()

        assert num_images == self.num_images_per_bag, "バッグの画像数が異なります！"

        x = x.view(batch_size * num_images, channels, height, width)  # バッチ内の画像としてまとめる（バッグの概念を外す）
        x = self.cnn(x)  # Process each image in the bag
        
        x = x.view(batch_size, num_images, self.input_dim)  # 画像の特徴量を１次元化する
        
        # Apply attention mechanism
        attended_x, attention_weights = self.attention(x)
        self.attention_weights = attention_weights
        
        # Fully connected layers
        x = F.relu(self.fc1(attended_x))
        x = self.fc2(x)
        x = x.squeeze(dim=-1)  # (batch_size,) に変換
        
        return x

    # Training step (パラメータの学習)
    def training_step(self, batch):
        # print('\n\n== TRAINING NOW!! ==\n\n')
        images, bag_labels = batch
        # outputs = self(images)  # 予測値 [batch_size, num_classes]
        outputs = self.forward(images)  # この段階で(batch_size,)にしなければならない

        # バギング内に0が含まれていれば、bag_labelは0、それ以外は1
        train_loss = F.binary_cross_entropy_with_logits(outputs, bag_labels.float())  # バイナリクロスエントロピー
        predictions = torch.sigmoid(outputs).round()
        train_acc = (predictions.squeeze() == bag_labels).float().mean()
        self.training_step_outputs.append({'train_loss': train_loss.detach(), 'train_acc': train_acc})

        values = {"loss": train_loss, "acc": train_acc}
        self.log_dict(values)

        loss = train_loss
        return loss
    
    # trainingのみなら以下の関数でlossを記録できる
    def on_train_epoch_end(self):
        all_outputs = self.training_step_outputs
        
        avg_loss = torch.stack([x['train_loss'] for x in all_outputs]).mean()
        avg_acc = torch.stack([x['train_acc'] for x in all_outputs]).mean()

        # self.training_step_outputs.clear()

        L = avg_loss.item()
        A = avg_acc.item()
        # print(f'\n\n== TRAIN RESULT ==\nval_loss : {L}\nval_acc : {A}\n\n')     

    # Validation step (ハイパーパラメータの学習、ここではloss&accuracyの管理のみ)
    def validation_step(self, batch, batch_idx):
        # print('\n\n== VALIDATION NOW!! ==\n\n')
        images, bag_labels = batch
        # outputs = self(images)  # 予測値 [batch_size, num_classes]
        outputs = self.forward(images)  # 0 ~ 1 の値がバッチ数分入っている
        
        val_loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), bag_labels.float())
        predictions = torch.sigmoid(outputs).round()  # 予測を0または1に丸める（四捨五入）
        val_acc = (predictions.squeeze() == bag_labels).float().mean()  # 正解率
        self.validation_step_outputs.append({'val_loss': val_loss.detach(), 'val_acc': val_acc})

        # return {'val_loss': val_loss, 'val_acc': val_acc}

    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in all_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in all_outputs]).mean()

        # self.validation_step_outputs.clear()

        L = avg_loss.item()
        A = avg_acc.item()
        print(f'\n== VALID RESULT ==\nval_loss : {L}\nval_acc : {A}\n')  
        
        # return {'val_loss': avg_loss, 'val_acc': avg_acc}

    # Optimizer setup
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        return optimizer
    
    def test_step(self, batch, batch_idx):
        images, bag_labels = batch
        outputs = self.forward(images)
        predictions = torch.sigmoid(outputs).round()
        val_acc = (predictions.squeeze() == bag_labels).float().mean()
        self.test_step_outputs.append(val_acc.item())
        
        # A = val_acc
        # print(f'\n\n== TEST RESULT ==\nacc : {A}\n\n')


""" モデルの初期化 """
# model = BaggedAttentionModel(num_images_per_bag=16)  #バッグの画像数は16!
datamodule = MyDataModule(batch_size=64, data_dir='./data', num_images_per_bag=16)
datamodule.set()


''' バッチの最初のバッグの画像を表示 '''
# visualizer = BagImageVisualizer(datamodule)
# visualizer.plot_bag(1)  # 1番目のバッグを表示
# visualizer.plot_bag(18) # 18番目のバッグを表示


""" training """
trainer = Trainer(
    max_epochs=10,
    # log_every_n_steps=10, 
    # enable_checkpointing=True,
    # profiler="simple"
)

# Train the model
# trainer.fit(model, datamodule)


""" 訓練課程のプロット """
# plotter = TrainingHistoryPlotter(model, save_flag=True)  # plot.py参照
# plotter.plot()


""" 以下ロードしたモデルを用いる """
model = BaggedAttentionModel.load_from_checkpoint(r"lightning_logs\version_103\checkpoints\epoch=9-step=420.ckpt")  
model.eval()
print(model.num_images_per_bag)


""" 精度の確認 """
# trainer.validate(model, dataloaders = datamodule.val_bagged_dataloader)
# trainer.test(model, dataloaders = datamodule.test_bagged_dataloader)
# result = model.test_step_outputs

# A = 0
# for i in result:
#     A = A + i
# final_acc = A / len(result)
# print(f"\nfinal accuracy : {final_acc}\n")


""" attention map """
visualizer = AttentionVisualizer(model, datamodule, bag_number=3, save_flag=False)  # plot.py参照
visualizer.visualize()
