import numpy as np

batch_size = 12
num_classes = 5
seq_len = 1000
output = np.random.rand(batch_size, seq_len, num_classes)
target = output + 1e-3 * np.random.rand(batch_size, seq_len, num_classes)
for itr in range(3):
    if itr == 0:
        cum_corr = np.zeros((batch_size, num_classes))
        valid_corr = np.zeros((batch_size, num_classes))
    for batch_idx in range(batch_size):
        for class_idx in range(num_classes):
        # compute correlation, apply fisher's transform
            corr = np.arctanh(np.corrcoef(target[batch_idx, :, class_idx].squeeze(),
                                          output[batch_idx, :, class_idx].squeeze())[0, 1])

            if not np.isnan(corr):
                cum_corr[batch_idx, class_idx] += corr
                valid_corr[batch_idx, class_idx] += 1
# average the correlations across over iterations apply inverse fisher's transform find mean over batch
    if num_classes == 1:
        avg_corr = np.tanh(cum_corr.squeeze() / valid_corr.squeeze()).mean()
    else:
        avg_corr = dict()
        for i in range(num_classes):
            avg_corr['Class%d' % i] = np.tanh(cum_corr[:, i] / valid_corr[:, i]).mean()

print(avg_corr)