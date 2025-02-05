''' バッチの最初のバッグの画像を表示 '''
import torch
import matplotlib.pyplot as plt
class BagImageVisualizer:
    def __init__(self, datamodule, num_images_per_bag=16):
        """
        :param datamodule: データモジュール (MNISTデータセットを使用することを前提)
        :param num_images_per_bag: 1つのバッグに含まれる画像の数
        """
        self.datamodule = datamodule
        self.num_images_per_bag = num_images_per_bag

    def get_bag_data(self, bag_index):
        """
        特定のバッグの画像とラベルを取得
        :param bag_index: バッグのインデックス
        :return: 画像テンソルとラベル
        """
        bag_index = bag_index - 1
        test_loader = self.datamodule.test_dataloader()
        data_iter = iter(test_loader)
        bag_images, labels = next(data_iter)
        return bag_images[bag_index], labels[bag_index]

    def plot_bag(self, bag_index):
        """
        指定されたバッグの画像をプロット
        :param bag_index: バッグのインデックス
        """
        # バッグの画像とラベルを取得
        bag, label = self.get_bag_data(bag_index)

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        axes = axes.flatten()

        # 画像を表示
        for i in range(self.num_images_per_bag):
            ax = axes[i]
            ax.imshow(bag[i].squeeze(0).cpu().numpy(), cmap='gray')
            ax.axis('off')

        classes = ["Does not contain zero", "Contains zero"]
        first_label = classes[label.item()]
        plt.suptitle(f"Bag {bag_index} : {first_label}")
        plt.show()


""" 学習結果の表示 """
import matplotlib.pyplot as plt
class TrainingHistoryPlotter:
    def __init__(self, model, save_dir="training_history", save_flag=False):
        """
        初期化: モデルのトレーニング履歴を取得
        :param model: トレーニング履歴が格納されているモデル
        :param save_dir: 保存先ディレクトリ
        :param save_flag: グラフを保存するかどうかを指定するフラッグ
        """
        self.train_history = model.training_step_outputs
        self.val_history = model.validation_step_outputs
        
        # 損失と精度をリストとして取り出す
        self.train_loss_list = [entry['train_loss'] for i, entry in enumerate(self.train_history) if i % 10 == 0]
        self.train_accuracy_list = [entry['train_acc'] for i, entry in enumerate(self.train_history) if i % 10 == 0]
        self.val_loss_list = [entry['val_loss'] for i, entry in enumerate(self.val_history) if i % 10 == 0]
        self.val_accuracy_list = [entry['val_acc'] for i, entry in enumerate(self.val_history) if i % 10 == 0]
        
        # エポック数
        self.train_epochs = range(1, len(self.train_loss_list) + 1)
        self.val_epochs = range(1, len(self.val_loss_list) + 1)

        # 保存フラッグと保存先ディレクトリ
        self.save_flag = save_flag
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot(self):
        """
        トレーニングとバリデーションの損失と精度をプロット
        """
        # グラフの作成
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 6))

        # Training LossとAccuracyのプロット
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train Loss', color='tab:blue')
        ax1.plot(self.train_epochs, self.train_loss_list, color='tab:blue', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()  # 同じx軸を使って別のy軸を作成
        ax2.set_ylabel('Train Accuracy', color='tab:orange')
        ax2.plot(self.train_epochs, self.train_accuracy_list, color='tab:orange', label='Train Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Validation LossとAccuracyのプロット
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Validation Loss', color='tab:blue')
        ax3.plot(self.val_epochs, self.val_loss_list, color='tab:blue', label='Validation Loss')
        ax3.tick_params(axis='y', labelcolor='tab:blue')
        ax4 = ax3.twinx()  # 同じx軸を使って別のy軸を作成
        ax4.set_ylabel('Validation Accuracy', color='tab:orange')
        ax4.plot(self.val_epochs, self.val_accuracy_list, color='tab:orange', label='Validation Accuracy')
        ax4.tick_params(axis='y', labelcolor='tab:orange')

        # レイアウトの調整
        fig.tight_layout()

        # グラフを表示
        if self.save_flag:
            self.save_plot(fig)  # 保存する場合は保存メソッドを呼び出す
        else:
            plt.show()

    def save_plot(self, fig):
        """
        グラフを指定したディレクトリに保存する
        :param fig: 保存するグラフのfigureオブジェクト
        """
        save_path = os.path.join(self.save_dir, "training_history_plot.png")
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Training history plot saved at: {save_path}")


""" attentionの表示 """
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
class AttentionVisualizer:
    def __init__(self, model, datamodule, bag_number=1, save_dir="attention_maps", save_flag=False):
        self.model = model
        self.datamodule = datamodule
        self.bag_number = bag_number
        self.classes = ["Does not contain zero", "Contains zero"]
        self.save_dir = save_dir
        self.save_flag = save_flag  # 保存フラッグを追加
        # 保存ディレクトリが存在しない場合は作成
        os.makedirs(self.save_dir, exist_ok=True)

    def get_data(self):
        # データローダーからデータを取得
        test_loader = self.datamodule.test_dataloader()
        data_iter = iter(test_loader)
        bag_images, labels = next(data_iter)
        return bag_images, labels

    def get_prediction_and_attention(self, bag_images, labels):
        # モデルを使って予測とアテンションマップを取得
        outputs = self.model.forward(bag_images)
        image = bag_images[self.bag_number]
        label = labels[self.bag_number]
        first_output = outputs[self.bag_number]
        prediction = torch.sigmoid(first_output).round().item()
        self.prediction = prediction

        attentions = self.model.attention_weights.detach()
        first_attention = attentions[self.bag_number]

        # Min-Max正規化
        tensor_min = torch.min(first_attention)
        tensor_max = torch.max(first_attention)
        normalized_attention = (first_attention - tensor_min) / (tensor_max - tensor_min)
        normalized_attention = normalized_attention.numpy()  # numpyに変換
        normalized_attention = first_attention.numpy()  # numpyに変換

        return prediction, first_attention, normalized_attention, image, label

    def plot_attention(self, image, normalized_attention, label):
        # アテンションマップを画像に重ねて表示
        fig, axes = plt.subplots(4, 4, figsize=(10, 7))

        for i, ax in enumerate(axes.flat):
            # 画像を表示
            img = image[i].squeeze().numpy()  # (1, 28, 28) の形なので squeeze して (28, 28) に
            ax.imshow(img, cmap='gray', interpolation='none')

            # Attentionマップをカラーマップに適用
            norm = mcolors.Normalize(vmin=0, vmax=1)
            cmap = plt.cm.viridis  # 例としてviridisカラーマップを使用

            overlay = np.ones_like(img) * normalized_attention[i]
            ax.imshow(overlay, cmap=cmap, norm=norm, alpha=0.5)
            ax.set_title(str(normalized_attention[i]))
            ax.axis('off')

        # カラーバーの追加（横に表示）
        cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])  # カラーバーの位置とサイズ
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)

        fig.subplots_adjust()
        plt.suptitle(f"Prediction of bag: {self.classes[int(self.prediction)]}\nGround truth: {self.classes[label]}")
        if self.save_flag:
            self.save_attention_map(fig)  # 保存フラッグがTrueの場合に画像を保存
        plt.show()

    def save_attention_map(self, fig):
        # ファイル名を指定して保存
        save_path = os.path.join(self.save_dir, f"attention_map_bag_{self.bag_number}.jpg")
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Attention map saved at: {save_path}")

    def visualize(self):
        # データを取得
        bag_images, labels = self.get_data()

        # 予測とアテンションを取得
        _, _, normalized_attention, image, label = self.get_prediction_and_attention(bag_images, labels)

        # 画像にアテンションマップを重ねて表示
        self.plot_attention(image, normalized_attention, label)