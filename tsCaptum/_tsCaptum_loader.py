from torch.utils.data import Dataset


class _tsCaptum_loader(Dataset):

	def __init__(self, X, labels):
		super().__init__()
		self.X = X
		self.labels = labels

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		return self.X[idx], self.labels[idx]
