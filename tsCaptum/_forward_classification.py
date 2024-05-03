def forward_classification(X_test : torch.Tensor, model):
	# convert X to pytorch tensor
	X_test_numpy = X_test.detach().numpy()
	# compute probability
	predictions = model.predict_proba(X_test_numpy)
	# return result as torch tensor as expected by captum attribution method
	return torch.tensor(predictions)

