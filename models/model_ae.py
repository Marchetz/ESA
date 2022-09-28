import torch
import torch.nn as nn
import pdb

class model_ae(nn.Module):
    """
    Encoder-Decoder model. The model reconstructs the future trajectory from an encoding of both past and future.
    Past and future trajectories are encoded separately.
    A trajectory is first convolved with a 1D kernel and are then encoded with a Gated Recurrent Unit (GRU).
    Encoded states are concatenated and decoded with a GRU and a fully connected layer.
    The decoding process decodes the trajectory step by step, predicting offsets to be added to the previous point.
    """
    def __init__(self, settings):
        super(model_ae, self).__init__()

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        input_gru = channel_out

        # temporal encoding
        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.conv_fut = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)

        # encoder-decoder
        self.encoder_past = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)
        self.encoder_fut = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)
        self.decoder = nn.GRU(self.dim_embedding_key * 3, self.dim_embedding_key * 3, 1, batch_first=False)
        self.FC_output = torch.nn.Linear(self.dim_embedding_key * 3, 2)

        # map encoding (CNN)
        # scene: input shape (batch, classes, 120, 120)
        self.convScene_1 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4))
        self.convScene_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8))
        self.fc_featScene = nn.Linear(15*15*8, self.dim_embedding_key)

        self.fc_upsampling = nn.Linear(self.dim_embedding_key * 3, 225)
        self.convScene_t1 = nn.Sequential(nn.ConvTranspose2d(1, 4, 3, stride=2,padding=1,output_padding=1), nn.ReLU(True))
        self.convScene_t2 = nn.Sequential(nn.ConvTranspose2d(4, 8, 3, stride=2,padding=1,output_padding=1), nn.ReLU(True))
        self.convScene_t3 = nn.Sequential(nn.ConvTranspose2d(8, 2, 3, stride=2,padding=1,output_padding=1))

        # activation and pooling function
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
        self.maxpool = nn.MaxPool2d(2)

        # weight initialization: kaiming
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.conv_fut.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)
        nn.init.kaiming_normal_(self.encoder_fut.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_fut.weight_hh_l0)
        nn.init.kaiming_normal_(self.decoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.decoder.weight_hh_l0)
        nn.init.kaiming_normal_(self.FC_output.weight)
        nn.init.kaiming_normal_(self.convScene_1[0].weight)
        nn.init.kaiming_normal_(self.convScene_2[0].weight)
        nn.init.kaiming_normal_(self.convScene_t1[0].weight)
        nn.init.kaiming_normal_(self.convScene_t2[0].weight)
        nn.init.kaiming_normal_(self.convScene_t3[0].weight)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.conv_fut.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)
        nn.init.zeros_(self.encoder_fut.bias_ih_l0)
        nn.init.zeros_(self.encoder_fut.bias_hh_l0)
        nn.init.zeros_(self.decoder.bias_ih_l0)
        nn.init.zeros_(self.decoder.bias_hh_l0)
        nn.init.zeros_(self.FC_output.bias)
        nn.init.zeros_(self.convScene_t1[0].bias)
        nn.init.zeros_(self.convScene_t2[0].bias)
        nn.init.zeros_(self.convScene_t3[0].bias)


    def forward(self, past, future, scene):
        """
        Forward pass that encodes past and future and decodes the future.
        :param past: past trajectory
        :param future: future trajectory
        :param scene: scene
        :return: decoded future, decoded scene
        """

        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key * 3)
        prediction = torch.Tensor()
        present = past[:, -1, :2].unsqueeze(1)
        if self.use_cuda:
            zero_padding = zero_padding.cuda()
            prediction = prediction.cuda()

        # temporal encoding for past
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)

        # temporal encoding for future
        future = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)

        # sequence encoding
        output_past, state_past = self.encoder_past(story_embed)
        output_fut, state_fut = self.encoder_fut(future_embed)

        # scene encoding
        scene_1 = self.convScene_1(scene) # 4,60,60
        scene_1_pool = self.maxpool(scene_1) # 4,30,30
        scene_2 = self.convScene_2(scene_1_pool) # 8,15,15
        feat_scene = self.fc_featScene(scene_2.reshape(-1, 15*15*8)) # 48
        feat_scene = nn.Tanh()(feat_scene)

        # encoding concatenation (TODO: provare senza dropout)
        state_conc = torch.cat((state_past, feat_scene.unsqueeze(0), state_fut), 2)
        #state_conc = self.dropout(state_conc)

        # future prediction (ae)
        input_dec = state_conc
        state_dec = zero_padding
        for i in range(self.future_len):
            output_decoder, state_dec = self.decoder(input_dec, state_dec)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present = coords_next
            input_dec = zero_padding

        # map prediction (ae)
        feat_up = self.fc_upsampling(state_conc).reshape(-1, 15, 15).unsqueeze(1)
        scene_t1 = self.convScene_t1(feat_up)
        scene_t2 = self.convScene_t2(scene_t1)
        scene_t3 = self.convScene_t3(scene_t2)
        return prediction, scene_t3
