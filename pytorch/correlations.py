import torch

def pearsonr(x, y):
	mean_x = torch.mean(x)
	mean_y = torch.mean(y)
	xm = x.sub(mean_x)
	ym = y.sub(mean_y)
	r_num = xm.dot(ym)
	#r_num = torch.dot(xm,ym)
	r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
	r_val = r_num / r_den
	return r_val

def linr(x, y):
	mean_x = torch.mean(x)
	mean_y = torch.mean(y)
	xm = x.sub(mean_x)
	ym = y.sub(mean_y)
	xy = torch.pow(x.sub(y),2.0)
	s_xy = xm.dot(ym)
	s_x2 = torch.norm(xm, 2)
	s_y2 = torch.norm(ym, 2)
	xy_2 = torch.mean(xy)
	r_val = 2.* s_xy / (s_x2 + s_y2 + xy_2)
	return r_val

def calculate_correlations(lin_corr, targets, *args):

	to_return = []

	for el in args:

		for i in range(targets.size()[1]):
			if lin_corr:
				corr = linr(targets[:,i], el[:,i])
			else:
				corr = pearsonr(targets[:,i], el[:,i])
			to_return.append(corr)

	return to_return

def calculate_correlations(lin_corr, targets, *args):

	to_return = []

	for el in args:

		for i in range(targets.size()[1]):
			if lin_corr:
				corr = linr(targets[:,i], el[:,i])
			else:
				corr = pearsonr(targets[:,i], el[:,i])
			to_return.append(corr)

	return to_return

def calculate_accuracies(labels, *args):

	to_return = []

	for el in args:
		acc = accuracy(el, labels)
		to_return.append(acc)

	return to_return

def discretize_labels(labels):
	labels[labels>0.5]=1
	labels[labels<=0.5]=0

	classes=torch.Tensor(labels.size()[0])

	for i, row in enumerate(labels):
		classes[i] = row[0]*4+row[1]*2+row[2]

	return classes.long()

def accuracy(output, labels):
	correct = 0
	total = 0

	_, predicted = torch.max(output, 1)
	total = labels.size(0)
	correct = (predicted == labels).sum()
	return (100 * correct / total)

if __name__ == '__main__':

	pearson = True

	t1 = torch.rand(800)
	t2 = torch.rand(800)

	print(pearsonr(t1,t2))
	print(linr(t1,t2))

	a = torch.rand(800,3)
	b = torch.rand(800,3)
	c = torch.rand(800,3)
	d = torch.rand(800,3)

	print(calculate_correlations((not pearson),a,b,c,d))

	e = torch.rand(800,8)
	f = torch.rand(800,8)
	g = torch.rand(800,8)
	h = torch.rand(800,3)

	print(calculate_accuracies(discretize_labels(h), e, f, g))

