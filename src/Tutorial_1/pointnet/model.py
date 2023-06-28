import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,N,k]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.

        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.mlp3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

    def forward(self, pointcloud):
        """
        Input:
            pointcloud: [B,N,3]
        Output:
            Global feature: [B,1024]
        """

        # TODO : Implement forward function.
        x = pointcloud
        x64 = x

        if self.input_transform:
            T = self.stn3(x.transpose(2, 1))
            x = torch.bmm(x, T)

        x = F.relu(self.mlp1(x.transpose(2, 1)))
        
        if self.feature_transform:
            T = self.stn64(x)
            x = torch.bmm(x.transpose(2, 1), T)
            x = x.transpose(2, 1)
            x64 = x
            

        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = torch.max(x, 2)[0]
        return x, x64


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - ...
        """
        # TODO : Implement forward function.
        pointcloud = pointcloud.to('cuda')
        feature, _ = self.pointnet_feat(pointcloud)
        output = self.fc(feature)
        output = F.log_softmax(output, dim=1)
        return output


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat(True, True)

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.mlp1 = nn.Sequential(nn.Linear(1024, int(num_points / 4)), nn.BatchNorm1d(int(num_points / 4)))
        self.mlp2 = nn.Sequential(nn.Linear(int(num_points / 4), int(num_points / 2)), nn.BatchNorm1d(int(num_points / 2)))
        self.mlp3 = nn.Sequential(nn.Linear(int(num_points / 2), num_points), nn.Dropout(p=0.2), nn.BatchNorm1d(num_points))
        self.mlp4 = nn.Linear(num_points, num_points * 3)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        pointcloud = pointcloud.to('cuda')
        feature, _ = self.pointnet_feat(pointcloud)

        x = feature

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = self.mlp4(x)

        B = x.shape[0]

        output = x.reshape(B, -1, 3)

        return output


class PointNetPartSeg(nn.Module):
    # TODO: Implement this
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes

        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        self.mlp1 = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512))
        self.mlp2 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256))
        self.mlp3 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128))
        self.mlp4 = nn.Sequential(nn.Conv1d(128, num_classes, 1), nn.BatchNorm1d(num_classes))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement this

        pointcloud = pointcloud.to('cuda')
        feature, x64 = self.pointnet_feat(pointcloud)

        #Shape of returned tensors
        #x64 [B, 64, N]
        #feature [B, 1024]

        dim2 = x64.shape[2]

        feature = feature.unsqueeze(2).expand(-1, -1, dim2) # [B, 1024, N]

        x = torch.cat([x64, feature], dim=1) #[M, N+N, K] [B, 1088, N]

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x)) # point features
        
        output = F.relu(self.mlp4(x))
        output = F.log_softmax(output, dim=1)

        return output

def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
