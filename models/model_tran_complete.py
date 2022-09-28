import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from models.multihead.activation import MultiheadAttention
import matplotlib.pyplot as plt
import time
import tqdm


class model_tran_complete(nn.Module):
    """
    Memory Network model with learnable writing controller.
    """

    def __init__(self, settings, model_pretrained, model_controller, model_IRM):
        super(model_tran_complete, self).__init__()
        self.name_model = 'model_tran_complete'
        self.mode = 'train' #train, val
        self.beta_value = []

        # parameters
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.num_prediction = settings["num_prediction"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]

        # Memory
        self.memory_past = torch.Tensor().cuda()
        self.memory_scene = torch.Tensor().cuda()
        self.memory_fut = torch.Tensor().cuda()
        self.memory_index = torch.Tensor()
        self.memory_track = torch.Tensor().cuda()


        # layers
        self.conv_past = model_pretrained.conv_past
        self.conv_fut = model_pretrained.conv_fut

        self.encoder_past = model_pretrained.encoder_past
        self.encoder_fut = model_pretrained.encoder_fut
        self.decoder = model_pretrained.decoder
        self.FC_output = model_pretrained.FC_output

        #scene
        self.convScene_1 = model_pretrained.convScene_1
        self.convScene_2 = model_pretrained.convScene_2
        self.fc_featScene = model_pretrained.fc_featScene

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.linear_controller = model_controller.linear_controller
        self.save_plot_weight()
        self.ablation_mode = None

        #IRM
        # scene: input shape (batch, classes, 360, 360)
        self.convScene_1_IRM = model_IRM.convScene_1
        self.convScene_2_IRM = model_IRM.convScene_2

        #self.RNN_scene = model_IRM.RNN_scene
        #self.model_IRM.RNN_scene.state_dict()
        self.RNN_scene = nn.GRU(32, self.dim_embedding_key, 1, batch_first=True)
        self.RNN_scene.bias_hh_l0 = model_IRM.RNN_scene.bias_hh_l0
        self.RNN_scene.bias_ih_l0 = model_IRM.RNN_scene.bias_ih_l0
        self.RNN_scene.weight_hh_l0 = model_IRM.RNN_scene.weight_hh_l0
        self.RNN_scene.weight_ih_l0 = model_IRM.RNN_scene.weight_ih_l0

        # refinement fc layer
        self.fc_refine = model_IRM.fc_refine

        embed_dim = 48
        num_heads = 1
        self.multihead_attn = nn.ModuleList([MultiheadAttention(embed_dim*2, num_heads, kdim=embed_dim*2, vdim=embed_dim) for i in range(self.num_prediction)])

        # embed_dim = 48
        # num_heads = 6
        # self.multihead_attn = MultiheadAttention(embed_dim*2, num_heads, kdim=embed_dim*2, vdim=embed_dim)


        self.decoder_layer = nn.TransformerDecoderLayer(d_model=144, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.fc_projector = nn.Linear(144,48)
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(embed_dim,200)
        self.linear2 = nn.Linear(200,embed_dim)

        self.maxpool = nn.MaxPool2d(2)

        #self.reset_parameters()


    def memory_float16(self):
        self.memory_past = self.memory_past.type(torch.float16).cuda()
        self.memory_fut = self.memory_fut.type(torch.float16).cuda()
        self.memory_scene = self.memory_scene.type(torch.float16).cuda()

    def encoder_scene(self, scene):
        scene_1 = self.convScene_1(scene) # 4,60,60
        scene_1_pool = self.maxpool(scene_1) # 4,30,30
        scene_2 = self.convScene_2(scene_1_pool) # 8,15,15
        feat_scene = self.fc_featScene(scene_2.reshape(-1, 15*15*8)) # 48
        feat_scene = nn.Tanh()(feat_scene)
        return feat_scene

    def encoder_track(self, track):
        track = torch.transpose(track, 1, 2)
        track_embed = self.relu(self.conv_past(track))
        track_embed = torch.transpose(track_embed, 1, 2)
        output, state = self.encoder_past(track_embed)
        return output, state

    def check_memory(self):
        """
        method to generate a future track from past-future feature read from a index location of the memory.
        :param past: index of the memory
        :return: predicted future
        """

        with torch.no_grad():
            mem_past = self.memory_past
            mem_scene = self.memory_scene
            mem_fut = self.memory_fut
            zero_padding = torch.zeros(1, len(mem_past), self.dim_embedding_key*3).cuda()
            present = torch.zeros(1, 2).cuda()
            prediction_single = torch.Tensor().cuda()
            info_total = torch.cat((mem_past, mem_scene, mem_fut), 1)
            input_dec = info_total.unsqueeze(0)
            state_dec = zero_padding
            for i in range(self.future_len):
                output_decoder, state_dec = self.decoder(input_dec, state_dec)
                displacement_next = self.FC_output(output_decoder)
                coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
                prediction_single = torch.cat((prediction_single, coords_next), 1)
                present = coords_next
                input_dec = zero_padding
            prediction_single = prediction_single.cpu()
            for pred in tqdm.tqdm(prediction_single):
                plt.plot(pred[:,0], pred[:,1])
            plt.axis('equal')
            plt.savefig('memory.png')
            plt.close()

    def init_memory(self, data_train):
        """
        Initialization: write element in memory.
        :param data_train: dataset
        :return: None
        """

        self.memory_past = torch.Tensor().cuda()
        self.memory_fut = torch.Tensor().cuda()
        self.memory_scene = torch.Tensor().cuda()
        self.memory_index = torch.Tensor()
        self.memory_track = torch.Tensor().cuda()


        #for count in range(10):
        j = random.randint(0, len(data_train)-1)
        idx, past, future, present, angle, video, vec, number_vec, scene, scene_onehot, scene_intention = data_train.__getitem__(j)

        past = torch.Tensor(past).unsqueeze(0).cuda()
        future = torch.Tensor(future).unsqueeze(0).cuda()
        scene_intention = torch.Tensor(scene_intention).unsqueeze(0).cuda()

        # encoding
        output_past, state_past = self.encoder_track(past)
        output_fut, state_fut = self.encoder_track(future)
        feat_scene = self.encoder_scene(scene_intention)

        state_past = state_past.squeeze(0)
        state_fut = state_fut.squeeze(0)

        self.memory_past = torch.cat((self.memory_past, state_past), 0).type(torch.float16)
        self.memory_scene = torch.cat((self.memory_scene, feat_scene), 0).type(torch.float16)
        self.memory_fut = torch.cat((self.memory_fut, state_fut), 0).type(torch.float16)
        self.memory_index = torch.cat((self.memory_index, torch.Tensor([int(idx)])), 0)
        #ablation study
        # future = torch.transpose(future, 1, 2)
        # self.memory_track = torch.cat((self.memory_track, future), 0)

    def forward_attn(self, past, state_past, feat_scene, info_future):
        #info_future: ex [6,256,48]
        # state_past: [1,256,48]

        # CONFIGURATION
        dim_batch = past.size()[0]
        present_temp = past[:, -1].unsqueeze(1).type(torch.float16)

        if self.memory_past.shape[0] < self.num_prediction:
            num_prediction = self.memory_past.shape[0]
        else:
            num_prediction = self.num_prediction
        prediction_single = torch.Tensor().type(torch.float16)
        if self.use_cuda:
            prediction_single = prediction_single.cuda()

        info_future = info_future.permute(1,0,2).reshape(-1,self.memory_past.shape[1]).unsqueeze(0)

        # DECODER
        present_temp = torch.repeat_interleave(present_temp, num_prediction, dim=0)
        state_past_repeat = torch.repeat_interleave(state_past, num_prediction, dim=1)
        state_scene_repeat = torch.repeat_interleave(feat_scene, num_prediction, dim=1)

        #pdb.set_trace()
        self.info_total = torch.cat((state_past_repeat, state_scene_repeat, info_future), 2)
        zero_padding = torch.zeros(1, self.info_total.shape[1], self.dim_embedding_key * 3).type(torch.float16)
        if self.use_cuda:
            zero_padding = zero_padding.cuda()

        input_dec = self.info_total
        state_dec = zero_padding
        for i in range(self.future_len):
            #pdb.set_trace()
            output_decoder, state_dec = self.decoder(input_dec, state_dec)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present_temp + displacement_next.squeeze(0).unsqueeze(1)
            prediction_single = torch.cat((prediction_single, coords_next), 1)
            present_temp = coords_next
            input_dec = zero_padding
        prediction = prediction_single.view(dim_batch, int(prediction_single.shape[0] / dim_batch), 30, 2)


        return prediction

    def forward(self, past, scene, scene_IRM, future=None, index=None):
        """
        Forward pass.
        Train phase: training writing controller based on reconstruction error of the future.
        Test phase: Predicts future trajectory based on past trajectory and the future feature read from the memory.
        :param past: past trajectory
        :param future: future trajectory (in test phase)
        :return: predicted future (test phase), writing probability and tolerance rate (train phase)
        """
        # CONFIGURATION
        dim_batch = past.size()[0]
        present_temp = past[:, -1].unsqueeze(1).type(torch.float16)
        out = []
        out_weight = []
        if self.memory_past.shape[0] < self.num_prediction:
            num_prediction = self.memory_past.shape[0]
        else:
            num_prediction = self.num_prediction
        prediction_single = torch.Tensor().type(torch.float16)
        if self.use_cuda:
            prediction_single = prediction_single.cuda()

        # ENCODING
        output_past, state_past = self.encoder_track(past)
        feat_scene = self.encoder_scene(scene)

        # MH ATTENTION
        query = torch.cat((state_past, feat_scene.unsqueeze(0)), 2)
        key = torch.cat((self.memory_past.unsqueeze(1).repeat(1, dim_batch, 1), self.memory_scene.unsqueeze(1).repeat(1, dim_batch, 1)), 2)
        value = self.memory_fut.unsqueeze(1).repeat(1, dim_batch, 1)
        #

        # out_single, attn_output_weights_single = self.multihead_attn(query, key, value)


        for i_m in range(num_prediction):
            out_single, attn_output_weights_single = self.multihead_attn[i_m](query, key, value)
            out_single = nn.Tanh()(out_single)
            #out_single = nn.Tanh()(self.linear2(self.relu(self.linear1(out_single))))
            out.append(out_single)
            out_weight.append(attn_output_weights_single)
        info_future = torch.cat(out).permute(1,0,2).reshape(-1,self.memory_past.shape[1]).unsqueeze(0)
        out_weight = torch.stack(out_weight).permute(1, 0, 2, 3)

        # DECODER
        present_temp = torch.repeat_interleave(present_temp, num_prediction, dim=0)
        state_past_repeat = torch.repeat_interleave(state_past, num_prediction, dim=1)
        state_scene_repeat = torch.repeat_interleave(feat_scene.unsqueeze(0), num_prediction, dim=1)

        # ABLATION
        if self.ablation_mode == 'past':
            state_past_repeat = torch.rand(state_past_repeat.shape).cuda()
        if self.ablation_mode == 'scene':
            state_scene_repeat = torch.rand(state_scene_repeat.shape).cuda()
        if self.ablation_mode == 'future':
            info_future = torch.rand(info_future.shape).cuda()
        self.info_total = torch.cat((state_past_repeat, state_scene_repeat, info_future), 2)
        zero_padding = torch.zeros(1, self.info_total.shape[1], self.dim_embedding_key * 3).type(torch.float16)
        if self.use_cuda:
            zero_padding = zero_padding.cuda()

        input_dec = self.info_total
        state_dec = zero_padding

        for i in range(self.future_len):
            output_decoder, state_dec = self.decoder(input_dec, state_dec)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present_temp + displacement_next.squeeze(0).unsqueeze(1)
            prediction_single = torch.cat((prediction_single, coords_next), 1)
            present_temp = coords_next
            input_dec = zero_padding

        if scene is not None:
            # scene encoding
            scene_IRM = scene_IRM.permute(0, 3, 1, 2)
            scene_1 = self.convScene_1_IRM(scene_IRM)
            scene_2 = self.convScene_2_IRM(scene_1)
            scene_2 = scene_2.repeat_interleave(num_prediction, dim=0)
            for i_refine in range(4):
                pred_map = prediction_single + 90
                pred_map = pred_map.unsqueeze(2)
                indices = pred_map.permute(0, 2, 1, 3)
                # rescale between -1 and 1
                indices = 2 * (indices / 180) - 1
                output = F.grid_sample(scene_2, indices, mode='nearest')
                output = output.squeeze(2).permute(0, 2, 1)

                state_rnn = state_past_repeat
                output_rnn, state_rnn = self.RNN_scene(output, state_rnn)
                prediction_refine = self.fc_refine(state_rnn).view(-1, self.future_len, 2)
                prediction_single = prediction_single + prediction_refine
        prediction = prediction_single.view(dim_batch, num_prediction, self.future_len, 2)

        #self.info_total = self.info_total.view()
        self.state_past = state_past.squeeze(0)
        self.feat_scene = feat_scene
        self.info_future = torch.cat(out).squeeze(0)
        # WRITING IN MEMORY
        if future is not None:
            future_rep = future.unsqueeze(1).repeat(1, num_prediction, 1, 1)
            distances = torch.norm(prediction - future_rep, dim=3)
            for step in range(future.shape[1]):
                # tolerance += distances[:, :, step] < 0.1 * (step + 1)
                tolerance_1s = torch.sum(distances[:, :, :10] < 0.5, dim=2)
                tolerance_2s = torch.sum(distances[:, :, 10:20] < 1, dim=2)
                tolerance_3s = torch.sum(distances[:, :, 20:30] < 1.5, dim=2)
                tolerance_4s = torch.sum(distances[:, :, 30:40] < 2, dim=2)
            tolerance = tolerance_1s + tolerance_2s + tolerance_3s + tolerance_4s
            tolerance_rate = torch.max(tolerance, dim=1)[0].type(torch.float16) / future.shape[1]
            tolerance_rate = tolerance_rate.unsqueeze(1).cuda()

            # controller
            writing_prob = torch.sigmoid(self.linear_controller(tolerance_rate))

            # future encoding
            # encoding
            output_fut, state_fut = self.encoder_track(future)

            #index = torch.Tensor(np.array(index).astype(np.int))
            index_writing = np.where(writing_prob.cpu() > 0.5)[0]
            past_to_write = state_past.squeeze()[index_writing]
            future_to_write = state_fut.squeeze()[index_writing]
            scene_to_write = feat_scene[index_writing]
            #index_to_write = index[index_writing]
            self.memory_past = torch.cat((self.memory_past, past_to_write), 0)
            self.memory_fut = torch.cat((self.memory_fut, future_to_write), 0)
            self.memory_scene = torch.cat((self.memory_scene, scene_to_write), 0)
            #self.memory_index = torch.cat((self.memory_index, index_to_write), 0)
            return prediction, out_weight, writing_prob, tolerance_rate
        else:
            return prediction, out_weight




    def save_plot_weight(self):

        x = torch.Tensor(np.linspace(0, 1, 100))
        weight = self.linear_controller.weight.cpu()
        bias = self.linear_controller.bias.cpu()


        y = torch.sigmoid(weight * x + bias).squeeze()
        plt.plot(x.data.numpy(), y.data.numpy(), '-r', label='y=' + str(weight.item()) + 'x + ' + str(bias.item()))
        #plt.plot(x.data.numpy(), y.data.numpy(), '-r', label='y=' + str(weight) + 'x + ' + str(bias))
        plt.plot(x.data.numpy(), [0.5] * 100, '-b')
        plt.title('controller')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='upper left')
        plt.grid()
        print('prova')


    def write_in_memory(self, past, future, scene, index):
        # TODO: change docs
        """
        Forward pass. Predicts future trajectory based on past trajectory and surrounding scene.
        :param past: past trajectory
        :param future: future trajectory
        :return: predicted future
        """

        if self.memory_past.shape[0] < self.num_prediction:
            num_prediction = self.memory_past.shape[0]
        else:
            num_prediction = self.num_prediction


        prediction, _, _, _ = self.forward(past, scene, future, index)

        # # TODO: change tolerance rate
        future_rep = future.unsqueeze(1).repeat(1, num_prediction, 1, 1)
        distances = torch.norm(prediction - future_rep, dim=3)
        for step in range(future.shape[1]):
            # tolerance += distances[:, :, step] < 0.1 * (step + 1)
            tolerance_1s = torch.sum(distances[:, :, :10] < 1.0, dim=2)
            tolerance_2s = torch.sum(distances[:, :, 10:20] < 1.5, dim=2)
            tolerance_3s = torch.sum(distances[:, :, 20:30] < 2.0, dim=2)
        tolerance = tolerance_1s + tolerance_2s + tolerance_3s
        tolerance_rate = torch.max(tolerance, dim=1)[0].type(torch.float32) / future.shape[1]
        tolerance_rate = tolerance_rate.unsqueeze(1).cuda()

        # controller
        writing_prob = torch.sigmoid(self.linear_controller(tolerance_rate))

        # future encoding
        # encoding
        output_fut, state_fut = self.encoder_track(future)

        index = torch.Tensor(np.array(index).astype(np.int))
        index_writing = np.where(writing_prob.cpu() > 0.5)[0]
        past_to_write = self.state_past.squeeze()[index_writing]
        future_to_write = state_fut.squeeze()[index_writing]
        scene_to_write = self.feat_scene[index_writing]
        index_to_write = index[index_writing]
        self.memory_past = torch.cat((self.memory_past, past_to_write), 0)
        self.memory_fut = torch.cat((self.memory_fut, future_to_write), 0)
        self.memory_scene = torch.cat((self.memory_scene, scene_to_write), 0)
        self.memory_index = torch.cat((self.memory_index, index_to_write), 0)

        # #ablation study: future track in memory
        # future = torch.transpose(future, 1, 2)
        # future_track_to_write = future[index_writing]
        # self.memory_track = torch.cat((self.memory_track,future_track_to_write), 0)




