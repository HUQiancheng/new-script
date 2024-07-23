import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from scripts.data.dataset import DSLRDataset
from scripts.data.transforms import ToTensor
from scripts.utils.load_save_models import save_checkpoint, get_latest_checkpoint, load_checkpoint
from scripts.networks.nn_2d_segformer import Segformer_Segmentation
from scripts.networks.nn_spconv import SimpleSpConvNet
from scripts.networks.utils_projection import project_to_3d, apply_softmax, colorize_point_cloud, save_point_cloud_to_ply, load_palette
from scripts.networks.utils_voxelization import PC2Tensor
from spconv.pytorch.utils import gather_features_by_pc_voxel_id

class Trainer:
    def __init__(self, config):
        self.config = config
        self._validate_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_environment()
        self.train_loader, self.val_loader = self._load_data()
        self.models, self.optimizers = self._initialize_models_and_optimizers()
        self.start_epoch = self._load_latest_checkpoints()
        self.writer = SummaryWriter(log_dir=self.config['paths']['tensorboard_log_dir'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config['training']['ignore_index'])

    def _validate_config(self):
        try:
            self.config['training']['learning_rate'] = float(self.config['training']['learning_rate'])
            self.config['training']['momentum'] = float(self.config['training']['momentum'])
            self.config['training']['weight_decay'] = float(self.config['training']['weight_decay'])
            self.config['training']['ignore_index'] = int(self.config['training']['ignore_index'])
        except ValueError as e:
            raise ValueError(f"Error in configuration values: {e}")

    def _setup_environment(self):
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(self.config['environment']['CUDA_LAUNCH_BLOCKING'])
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = self.config['environment']['PYTORCH_CUDA_ALLOC_CONF']
        torch.cuda.empty_cache()

    def _load_data(self):
        transform = T.Compose([ToTensor()])
        train_dataset = DSLRDataset(self.config['paths']['data_dir'], self.config['paths']['train_split_file'], transform=transform)
        val_dataset = DSLRDataset(self.config['paths']['data_dir'], self.config['paths']['val_split_file'], transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True, num_workers=self.config['training']['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False, num_workers=self.config['training']['num_workers'])
        return train_loader, val_loader

    def _initialize_models_and_optimizers(self):
        model_2d = Segformer_Segmentation(num_classes=self.config['model']['num_classes']).to(self.device)
        model_2d3d = SimpleSpConvNet(input_channels=self.config['model']['num_classes'], num_classes=self.config['model']['num_classes']).to(self.device)
        model_3d = SimpleSpConvNet(input_channels=1, num_classes=self.config['model']['num_classes']).to(self.device)
        optimizer_2d = optim.SGD(model_2d.parameters(), lr=self.config['training']['learning_rate'], momentum=self.config['training']['momentum'], weight_decay=self.config['training']['weight_decay'])
        optimizer_2d3d = optim.SGD(model_2d3d.parameters(), lr=self.config['training']['learning_rate'], momentum=self.config['training']['momentum'], weight_decay=self.config['training']['weight_decay'])
        optimizer_3d = optim.SGD(model_3d.parameters(), lr=self.config['training']['learning_rate'], momentum=self.config['training']['momentum'], weight_decay=self.config['training']['weight_decay'])
        return [model_2d, model_2d3d, model_3d], [optimizer_2d, optimizer_2d3d, optimizer_3d]

    def _load_latest_checkpoints(self):
        start_epochs = []
        for model, optimizer, model_name in zip(self.models, self.optimizers, ['2d', '2d3d', '3d']):
            checkpoint_dir = os.path.join(self.config['paths']['checkpoint_dir'], model_name)
            latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
            start_epoch = 0
            if latest_checkpoint:
                start_epoch, _ = load_checkpoint(latest_checkpoint, model, optimizer)
            start_epochs.append(start_epoch)
        return min(start_epochs)

    def calculate_accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        valid_mask = labels != self.config['training']['ignore_index']
        correct = torch.sum(preds[valid_mask] == labels[valid_mask]).item()
        return correct / valid_mask.sum().item()

    def process_batch(self, batch):
        inputs = batch['image'].to(self.device)
        labels_pc = batch['label'].to(self.device)
        depth = batch['depth'].to(self.device)
        R = batch['R'].to(self.device)
        T = batch['T'].to(self.device)
        K = batch['intrinsic_mat'].to(self.device)
        inputs1, inputs2 = torch.split(inputs, 1, dim=1)
        inputs1, inputs2 = inputs1.squeeze(1), inputs2.squeeze(1)
        labels1, labels2 = torch.split(labels_pc, 1, dim=1)
        labels1, labels2 = labels1.squeeze(1), labels2.squeeze(1)
        depth1, depth2 = torch.split(depth, 1, dim=1)
        depth1, depth2 = depth1.squeeze(1), depth2.squeeze(1)
        R1, R2 = torch.split(R, 1, dim=1)
        R1, R2 = R1.squeeze(1), R2.squeeze(1)
        T1, T2 = torch.split(T, 1, dim=1)
        T1, T2 = T1.squeeze(1), T2.squeeze(1)
        K1, K2 = torch.split(K, 1, dim=1)
        K1, K2 = K1.squeeze(1), K2.squeeze(1)
        invalid_mask1 = (labels1 >= 100) | (labels1 < 0)
        invalid_mask2 = (labels2 >= 100) | (labels2 < 0)
        labels1[invalid_mask1], labels2[invalid_mask2] = self.config['training']['ignore_index'], self.config['training']['ignore_index']
        return inputs1, inputs2, labels1, labels2, depth1, depth2, R1, R2, T1, T2, K1, K2

    def visualize_point_cloud(self, epoch, coords_pc, output_features, output_dir):
        output_pc = torch.cat((coords_pc, output_features.unsqueeze(0)), dim=2)
        pc_pred = apply_softmax(output_pc)
        palette = load_palette('scripts/utils/palette_scannet200.txt')
        output_pc_color = colorize_point_cloud(pc_pred, palette)
        save_point_cloud_to_ply(epoch, output_pc_color, output_dir)

    def train_epoch(self, epoch):
        self.models[0].train()
        self.models[1].train()
        self.models[2].train()

        running_loss_2d = 0.0
        running_loss_3d = 0.0
        running_corrects1 = 0
        running_corrects2 = 0
        running_corrects_3d = 0
        total_samples_2d = 0
        total_samples_3d = 0

        for i, batch in enumerate(self.train_loader):
            inputs1, inputs2, labels1, labels2, depth1, depth2, R1, R2, T1, T2, K1, K2 = self.process_batch(batch)
            self.optimizers[0].zero_grad()
            outputs1 = self.models[0](inputs1)
            outputs2 = self.models[0](inputs2)
            loss2d_1 = self.criterion(outputs1, labels1)
            loss2d_2 = self.criterion(outputs2, labels2)
            loss2d_1.backward()
            self.optimizers[0].step()
            loss2d = (loss2d_1 + loss2d_2) / 2
            running_loss_2d += loss2d.item()
            corrects1 = self.calculate_accuracy(outputs1, labels1)
            running_corrects1 += corrects1 * inputs1.size(0)
            corrects2 = self.calculate_accuracy(outputs2, labels2)
            running_corrects2 += corrects2 * inputs2.size(0)
            running_corrects = (running_corrects1 + running_corrects2) / 2
            total_samples_2d += inputs1.size(0)

            point_cloud_features1 = project_to_3d(outputs1, depth1, K1, R1, T1, ignore_index=self.config['training']['ignore_index'])
            point_cloud_features2 = project_to_3d(outputs2, depth2, K2, R2, T2, ignore_index=self.config['training']['ignore_index'])
            pc_2d = torch.cat((point_cloud_features1, point_cloud_features2), dim=1)
            coords_pc = batch['coord_pc'].to(self.device)
            labels_pc = batch['label_pc'].to(self.device).unsqueeze(-1)
            invalid_mask = (labels_pc < 0)
            labels_pc[invalid_mask] = self.config['training']['ignore_index']
            pc_3d = torch.cat((coords_pc, torch.ones(coords_pc.shape[0], coords_pc.shape[1], 1).to(self.device)), dim=-1)
            pc_3d_label = torch.cat((coords_pc, labels_pc), dim=-1)
            tensor_label = PC2Tensor(self.device, self.config['model']['spatial_shape'], use_label=True)
            tensor_pc = PC2Tensor(self.device, self.config['model']['spatial_shape'])
            input_3d, pc_voxel_id = tensor_label(pc_3d_label)
            input_2d = tensor_pc(pc_2d)
            self.optimizers[1].zero_grad()
            self.optimizers[2].zero_grad()
            output_2d = self.models[1](input_2d)
            output_3d = self.models[2](input_3d)
            output_2d_dense = output_2d.dense()
            output_3d_dense = output_3d.dense()
            output_dense = output_2d_dense + output_3d_dense
            for j in range(output_3d.indices.shape[0]):
                b, z, y, x = output_3d.indices[j]
                output_3d.features[j] = output_dense[b, :, z, y, x]
            output_features = gather_features_by_pc_voxel_id(output_3d.features, pc_voxel_id.squeeze(0))
            loss3d = self.criterion(output_features, labels_pc.squeeze(0).squeeze(-1))
            loss3d.backward()
            self.optimizers[1].step()
            self.optimizers[2].step()
            if epoch % 50 == 0:
                self.visualize_point_cloud(epoch, coords_pc, output_features, output_dir='./outputs/train')
            running_loss_3d += loss3d.item()
            corrects3d = self.calculate_accuracy(output_features, labels_pc.squeeze(0).squeeze(-1))
            running_corrects_3d += corrects3d * coords_pc.size(0)
            total_samples_3d += coords_pc.size(0)

        epoch_loss_2d = running_loss_2d / len(self.train_loader)
        epoch_acc_2d = running_corrects / total_samples_2d
        epoch_loss_3d = running_loss_3d / len(self.train_loader)
        epoch_acc_3d = running_corrects_3d / total_samples_3d
        return epoch_loss_2d, epoch_acc_2d, epoch_loss_3d, epoch_acc_3d

    def validate_model(self, epoch):
        self.models[0].eval()
        self.models[1].eval()
        self.models[2].eval()
        running_loss_2d = 0.0
        running_loss_3d = 0.0
        running_corrects1 = 0
        running_corrects2 = 0
        running_corrects_3d = 0
        total_samples_2d = 0
        total_samples_3d = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                inputs1, inputs2, labels1, labels2, depth1, depth2, R1, R2, T1, T2, K1, K2 = self.process_batch(batch)
                outputs1 = self.models[0](inputs1)
                outputs2 = self.models[0](inputs2)
                loss2d_1 = self.criterion(outputs1, labels1)
                loss2d_2 = self.criterion(outputs2, labels2)
                loss2d = (loss2d_1 + loss2d_2) / 2
                running_loss_2d += loss2d.item()
                corrects1 = self.calculate_accuracy(outputs1, labels1)
                running_corrects1 += corrects1 * inputs1.size(0)
                corrects2 = self.calculate_accuracy(outputs2, labels2)
                running_corrects2 += corrects2 * inputs2.size(0)
                running_corrects = (running_corrects1 + running_corrects2) / 2
                total_samples_2d += inputs1.size(0)
                point_cloud_features1 = project_to_3d(outputs1, depth1, K1, R1, T1, ignore_index=self.config['training']['ignore_index'])
                point_cloud_features2 = project_to_3d(outputs2, depth2, K2, R2, T2, ignore_index=self.config['training']['ignore_index'])
                pc_2d = torch.cat((point_cloud_features1, point_cloud_features2), dim=1)
                coords_pc = batch['coord_pc'].to(self.device)
                labels_pc = batch['label_pc'].to(self.device).unsqueeze(-1)
                invalid_mask = (labels_pc < 0)
                labels_pc[invalid_mask] = self.config['training']['ignore_index']
                pc_3d = torch.cat((coords_pc, torch.ones(coords_pc.shape[0], coords_pc.shape[1], 1).to(self.device)), dim=-1)
                pc_3d_label = torch.cat((coords_pc, labels_pc), dim=-1)
                tensor_label = PC2Tensor(self.device, self.config['model']['spatial_shape'], use_label=True)
                tensor_pc = PC2Tensor(self.device, self.config['model']['spatial_shape'])
                input_3d, pc_voxel_id = tensor_label(pc_3d_label)
                input_2d = tensor_pc(pc_2d)
                output_2d = self.models[1](input_2d)
                output_3d = self.models[2](input_3d)
                output_2d_dense = output_2d.dense()
                output_3d_dense = output_3d.dense()
                output_dense = output_2d_dense + output_3d_dense
                for j in range(output_3d.indices.shape[0]):
                    b, z, y, x = output_3d.indices[j]
                    output_3d.features[j] = output_dense[b, :, z, y, x]
                output_features = gather_features_by_pc_voxel_id(output_3d.features, pc_voxel_id.squeeze(0))
                loss3d = self.criterion(output_features, labels_pc.squeeze(0).squeeze(-1))
                running_loss_3d += loss3d.item()
                corrects3d = self.calculate_accuracy(output_features, labels_pc.squeeze(0).squeeze(-1))
                running_corrects_3d += corrects3d * coords_pc.size(0)
                total_samples_3d += coords_pc.size(0)

                # Save visualization for validation
                self.visualize_point_cloud(epoch, coords_pc, output_features, output_dir='./outputs/val')

        epoch_loss_2d = running_loss_2d / len(self.val_loader)
        epoch_acc_2d = running_corrects / total_samples_2d
        epoch_loss_3d = running_loss_3d / len(self.val_loader)
        epoch_acc_3d = running_corrects_3d / total_samples_3d

        # Write validation loss to TensorBoard
        self.writer.add_scalar('Validation Loss 2D', epoch_loss_2d, epoch)
        self.writer.add_scalar('Validation Accuracy 2D', epoch_acc_2d, epoch)
        self.writer.add_scalar('Validation Loss 3D', epoch_loss_3d, epoch)
        self.writer.add_scalar('Validation Accuracy 3D', epoch_acc_3d, epoch)

        return epoch_loss_2d, epoch_acc_2d, epoch_loss_3d, epoch_acc_3d

    def train(self, use_val=False):
        if use_val:
            checkpoint_dir = self.config['paths']['checkpoint_dir']
            for epoch in range(10, 90, 10):
                model_2d_checkpoint = os.path.join(checkpoint_dir, '2d', f'checkpoint_epoch_{epoch}.pth.tar')
                model_2d3d_checkpoint = os.path.join(checkpoint_dir, '2d3d', f'checkpoint_epoch_{epoch}.pth.tar')
                model_3d_checkpoint = os.path.join(checkpoint_dir, '3d', f'checkpoint_epoch_{epoch}.pth.tar')

                self.models[0].load_state_dict(torch.load(model_2d_checkpoint)['model_state_dict'])
                self.models[1].load_state_dict(torch.load(model_2d3d_checkpoint)['model_state_dict'])
                self.models[2].load_state_dict(torch.load(model_3d_checkpoint)['model_state_dict'])

                val_loss_2d, val_acc_2d, val_loss_3d, val_acc_3d = self.validate_model(epoch)
                print(f"Validation Loss 2D: {val_loss_2d}, Accuracy: {val_acc_2d}")
                print(f"Validation Loss 3D: {val_loss_3d}, Accuracy: {val_acc_3d}\n")
        else:
            for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
                epoch_loss_2d, epoch_acc_2d, epoch_loss_3d, epoch_acc_3d = self.train_epoch(epoch)
                print(f"2D Epoch [{epoch+1}/{self.config['training']['num_epochs']}], Loss: {epoch_loss_2d}, Accuracy: {epoch_acc_2d}")
                print(f"3D Epoch [{epoch+1}/{self.config['training']['num_epochs']}], Loss: {epoch_loss_3d}, Accuracy: {epoch_acc_3d}\n")

                self.writer.add_scalar('Training Loss 2D', epoch_loss_2d, epoch)
                self.writer.add_scalar('Training Accuracy 2D', epoch_acc_2d, epoch)
                self.writer.add_scalar('Training Loss 3D', epoch_loss_3d, epoch)
                self.writer.add_scalar('Training Accuracy 3D', epoch_acc_3d, epoch)

                if (epoch + 1) % self.config['training']['save_interval'] == 0 or (epoch + 1) == self.config['training']['num_epochs']:
                    save_checkpoint(epoch + 1, self.models[0], self.optimizers[0], epoch_loss_2d, os.path.join(self.config['paths']['checkpoint_dir'], '2d'), filename=f'checkpoint_epoch_{epoch+1}.pth.tar')
                    save_checkpoint(epoch + 1, self.models[1], self.optimizers[1], epoch_loss_2d, os.path.join(self.config['paths']['checkpoint_dir'], '2d3d'), filename=f'checkpoint_epoch_{epoch+1}.pth.tar')
                    save_checkpoint(epoch + 1, self.models[2], self.optimizers[2], epoch_loss_2d, os.path.join(self.config['paths']['checkpoint_dir'], '3d'), filename=f'checkpoint_epoch_{epoch+1}.pth.tar')

            print("Training complete")
            self.writer.close()

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    trainer = Trainer(config)
    trainer.train(use_val=False)  # Set use_val=True for validation mode
