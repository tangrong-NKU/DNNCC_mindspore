# coding:utf-8
from my_function import *
def cos_similarity(A):
    B = np.zeros((A.shape[0], A.shape[0]))
    for i in tqdm(range(B.shape[0])):
        for j in range(B.shape[1]):
            fenzi = np.sum(np.multiply(A[i], A[j]))
            fenmu = np.sum(np.multiply(A[i], A[i]))*np.sum(np.multiply(A[j], A[j]))
            if fenmu != 0:
                B[i, j] = fenzi/np.sqrt(fenmu)
            else:
                B[i, j] = 0
    return B

xp = np.loadtxt('../protein sequence coding model/datasets/DTINet/protein_embeds.csv', delimiter=',')
sp = cos_similarity(xp)
np.savetxt("sp.txt",sp)