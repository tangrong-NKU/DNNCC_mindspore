# coding:utf-8
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from my_function import *
import mindspore.dataset as ds
from mindspore import Tensor, set_context, PYNATIVE_MODE, dtype as mstype
import math


EPOCH = 100
LR = 0.0001
FOLD = 5
CUDA_AVAILABLE = 1
DRUG = 708
PROTEIN = 1512
set_context(mode=PYNATIVE_MODE)
# zero_cat = ops.Concat()
# one_cat = ops.Concat(1)
class MLP_CNN(nn.Cell):
    def __init__(self,size_r,size_p,att_size,fun_size):
       super(MLP_CNN,self).__init__()
       self.conv_l1 = nn.SequentialCell(
           nn.Conv2d(
               in_channels=1, # input height
               out_channels=16,  # n_filter
               kernel_size=(3, 5),  # 6*260->4*256
               pad_mode="pad",
               padding=2  # con2d出来的图片大小不变   2*256->6*260
                ),  # output shape [1, 16, 4, 256]
           nn.LeakyReLU(),
           # nn.MaxPool2d(2),  # [1, 1, 4, 256] -> [1, 1, 2, 128]
           nn.AvgPool2d(2)
       )

       self.conv_l2 = nn.SequentialCell(
           nn.Conv2d(in_channels=16,  # input height
                     out_channels=32,  # n_filter
                     kernel_size=(3, 5),  # filter size
                     pad_mode="pad",
                     padding=2  # con2d出来的图片大小不变
                     ),  # output shape [1, 32, 4, 128]
           nn.LeakyReLU(),
           nn.MaxPool2d(2),  # [1, 32, 4, 128] -> [1, 32, 2, 64]
       )

       # self.out = nn.Dense(int((att_size + fun_size) / 4 * 2 * 32), 2)
       self.out = nn.Dense(196352, 2)

       self.encoder_fun = nn.SequentialCell(
           nn.Dense(5603, 4096),
           nn.GELU(),
           nn.Dense(4096, fun_size),
           nn.Sigmoid()
       )

       self.encoder_att = nn.SequentialCell(
           nn.Dense(size_r, 2048),
           nn.GELU(),
           nn.Dense(2048, att_size),
           nn.Sigmoid()
       )


    def construct(self, r_att, p_att, r_fun, p_fun, idx):
        # print(1)
        # att = zero_cat((r_att,p_att))
        # print(type(r_att))
        att = ops.concat((r_att,p_att))
        # att = att.asnumpy()
        # print(type(att))
        # print(att)
        e_att = self.encoder_att(att)

        fun = ops.concat((r_fun, p_fun))
        e_fun = self.encoder_fun(fun)
        e = ops.concat((e_att, e_fun),1)
        # r_fun = ms.Tensor(r_fun)
        e_r = e[0: r_fun.shape[0], :]
        e_p = e[r_fun.shape[0]:, :]

        x_cnn = []
        # print(idx)
        for i in range(len(idx)):
            r_no = int(idx[i] / PROTEIN)
            p_no = int(idx[i] % PROTEIN)
            x_cnn.append(ops.concat((e_r[r_no, :], e_p[p_no, :])))
        # print(x_cnn[0])
        # print(type(x_cnn))
        # print(len(x_cnn))
        # x_cnn = x_cnn.asnumpy().tolist()
        x_cnn = ops.concat(x_cnn)


        # if CUDA_AVAILABLE == 1:
        #     x_cnn = x_cnn.cuda()

        x_cnn = x_cnn.view(-1, 1, 2, e.shape[1])
        embedding_cnn = self.conv_l1(x_cnn)  # [1, 1, 2, 256] -> [1, 16, 2, 128]
        embedding_cnn = self.conv_l2(embedding_cnn)  # [1, 16, 2, 128] -> [1, 32, 2, 64]
        # print(type(embedding_cnn))
        # embedding_cnn = nn.Tanh(embedding_cnn)
        # print(type(embedding_cnn))
        embedding_cnn = Tensor(embedding_cnn,ms.float32)
        tanh = nn.Tanh()
        embedding_cnn = tanh(embedding_cnn)
        # embedding_cnn = ms.Tensor(np.array(embedding_cnn))

        b, n_f, h, w = embedding_cnn.shape
        # print(embedding_cnn.shape)
        out1 = embedding_cnn.view((b, n_f * h * w))

        # print("out1:",out1.shape)
        output = self.out(out1)
        output = Tensor(output,ms.float32)
        # print(output.shape)
        return e_r, e_p, output
        # return e_r, e_p,x_cnn

# def cosine_sim(e):
#     l2_1 = e*(e.T)

# 损失网络
class MyWithLossCell(nn.Cell):
    def __init__(self,backbone,loss_fn):
        """实例化时传入前向网络和损失函数作为参数"""
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_func = loss_fn
    def construct(self, batch_y,r_att, p_att, r_fun, p_fun, idx):
        e_r, e_p, output = self.backbone(r_att, p_att, r_fun, p_fun, idx)
        loss = self.loss_func(output,batch_y)
        return loss

    def backbone_network(self):
        """要封装的骨干网络"""
        return self.backbone




def model_train(BATCH_SIZE, ATT_SIZE, FUN_SIZE):
    # 读取数据
    print('读取数据')
    A = np.loadtxt('../dataset/mat_drug_protein.txt')
    x1 = np.loadtxt('../dataset/RRI.txt')
    x2 = np.loadtxt('../dataset/RPI.txt')
    x3 = np.loadtxt('../dataset/RDI.txt')
    sr = np.loadtxt('../dataset/Similarity_Matrix_Drugs.txt')
    xr_m = sr.dot(np.hstack((x1, x2)))
    xr_m = np.array(xr_m,dtype=float32)
    # print(type(xr_m[0,2]))
    for i in range(xr_m.shape[0]):
        for j in range(xr_m.shape[1]):
            # print(i,j)
            if xr_m[i,j]<= 0.5:
                xr_m[i,j] = 0
    xr_c = x3  # 708*6576

    x4 = np.loadtxt('../dataset/PRI.txt')
    x5 = np.loadtxt('../dataset/PPI.txt')
    x6 = np.loadtxt('../dataset/PDI.txt')
    xp = np.loadtxt('../protein sequence coding model/datasets/DTINet/protein_embeds.csv', delimiter=',')
    # sp = cos_similarity(xp)
    sp = np.loadtxt("sp.txt")
    xp_m = sp.dot(np.hstack((x4, x5)))
    for i in range(xp_m.shape[0]):
        for j in range(xp_m.shape[1]):
            if xp_m[i, j] <= 0.5:
                xp_m[i, j] = 0
    xp_c = x6  # 1512*6576

    index_0 = np.loadtxt('result/DTI_index_0.txt')
    index_1 = np.loadtxt('result/DTI_index_1.txt')

    print('初始化网络')

    for f in range(1):
        # 4份1 共1536个
        fold_index_1 = index_1[0:f, :].flatten().tolist() + index_1[f + 1:FOLD, :].flatten().tolist()
        train_index_0 = index_0[0:f, :].flatten().tolist() + index_0[f + 1:FOLD, :].flatten().tolist()
        num_1 = len(fold_index_1)
        num_0 = len(train_index_0)
        train_index_1 = []
        while num_1 < num_0:
            train_index_1 += fold_index_1
            num_1 = len(train_index_1)
        # print(num_1, num_0)
        # train_index_1 += train_index_1
        train_index_1 = train_index_1[0: num_0]
        # train_x = []
        # train_y = []
        train_index = train_index_1+train_index_0

        # if CUDA_AVAILABLE == 1:
        #     model = model.cuda()
        model = MLP_CNN(size_r=xr_m.shape[1], size_p=xp_m.shape[1], att_size=ATT_SIZE, fun_size=FUN_SIZE)
        loss_func1 = nn.CrossEntropyLoss()
        loss_func2 = nn.MSELoss()

        dataloader = ds.NumpySlicesDataset(data=train_index, shuffle=True)
        dataloader = dataloader.batch(BATCH_SIZE).repeat(EPOCH)

        optimizer = nn.Adam(params=model.trainable_params(),learning_rate=LR)
        #定义模型
        net_with_criterion = MyWithLossCell(model,loss_func1)
        #构建训练网络
        train_net = nn.TrainOneStepCell(net_with_criterion,optimizer)

        for i,data in tqdm(enumerate(dataloader.create_dict_iterator())):
            batch_id = data['column_0']
            # print(batch_id.shape)
            xr_m_tensor = ms.Tensor(np.array(xr_m).astype(np.float32))
            xp_m_tensor = ms.Tensor(np.array(xp_m).astype(np.float32))
            xr_c_tensor = ms.Tensor(np.array(xr_c).astype(np.float32))
            xp_c_tensor = ms.Tensor(np.array(xp_c).astype(np.float32))
            batch_y = []
            # print(type(batch_id))
            # print(len(batch_idx))
            for i in range(len(batch_id)):
                drug_no = int(batch_id[i] / PROTEIN)
                protein_no = int(batch_id[i] % PROTEIN)
                batch_y.append(A[int(drug_no), int(protein_no)])
            batch_id = batch_id.asnumpy().tolist()
            # batch_idx = ms.Tensor(np.array(batch_idx,dtype=int_))
            batch_y = ms.Tensor(np.array(batch_y),ms.int32)
            train_net(batch_y,xr_m_tensor, xp_m_tensor, xr_c_tensor, xp_c_tensor, batch_id)
            # print(xr_c_tensor.shape
            loss = net_with_criterion(batch_y,xr_m_tensor, xp_m_tensor, xr_c_tensor, xp_c_tensor, batch_id)
            # print(loss)
            print('Epoch:',i, ' train loss: %.16f' % loss)



            # print(loss)
            # print('train loss: %.16f' % loss[0][0])






        # for epoch in range(EPOCH):
        #     for step, data in enumerate(dataloader):
        #         batch_idx = data[0]
        #         xr_m_tensor = ms.Tensor(np.array(xr_m).astype(np.float32))
        #         xp_m_tensor = ms.Tensor(np.array(xp_m).astype(np.float32))
        #         xr_c_tensor = ms.Tensor(np.array(xr_c).astype(np.float32))
        #         xp_c_tensor = ms.Tensor(np.array(xp_c).astype(np.float32))
        #         batch_y = []
        #         # print(len(batch_idx))
        #         batch_idx = batch_idx.asnumpy()
        #         for i in range(len(batch_idx)):
        #             drug_no = int(batch_idx[i] / PROTEIN)
        #             protein_no = int(batch_idx[i] % PROTEIN)
        #             batch_y.append(A[int(drug_no), int(protein_no)])
        #
        #         batch_idx = batch_idx.tolist()
        #         # batch_idx = ms.Tensor(np.array(batch_idx,dtype=int_))
        #         batch_y = ms.Tensor(np.array(batch_y))
        #
        #         # if CUDA_AVAILABLE == 1:
        #         #     batch_idx = batch_idx.cuda()
        #         #     batch_y = batch_y.cuda()
        #         #     xr_m_tensor = xr_m_tensor.cuda()
        #         #     xp_m_tensor = xp_m_tensor.cuda()
        #         #     xr_c_tensor = xr_c_tensor.cuda()
        #         #     xp_c_tensor = xp_c_tensor.cuda()
        #         # print(batch_idx[5])
        #
        #
        #
        #
        #
        #         # loss1 = loss_func1(out, batch_y)
        #         # loss = loss1
        #         # optimizer.zero_grad()
        #         # loss.backward()
        #         # optimizer.step()
        #         #
        #         # if step % 10 == 0:
        #         #     if CUDA_AVAILABLE == 1:
        #         #         print('FOLD:', f, 'Epoch: ', epoch, 'Item: ', step, math.ceil(len(train_index_1) / BATCH_SIZE),
        #         #               '| loss: %.20f' % loss1.cpu().item())
        #         #     else:
        #         #         print('FOLD:', f, 'Epoch: ', epoch, 'Item: ', step, ' | ',
        #         #               math.ceil(len(train_index_1) / BATCH_SIZE),
        #         #               '| loss: %.20f' % loss1.item())
        # para = str(BATCH_SIZE) + str(ATT_SIZE) + str(FUN_SIZE)
        # # ms.save(model, 'model/fold_' + str(f) + '_' + str(para) + '.pkl')

model_train(256,512,1024)