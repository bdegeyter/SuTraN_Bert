import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from SuTraN.transformer_prefix_encoder import EncoderLayer
from SuTraN.transformer_suffix_decoder import DecoderLayer, DecoderLayerCached

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """Inject sequence information in the prefix or suffix embeddings 
    before feeding them to the stack of encoders or decoders respectively. 

    Predominantly based on the PositionalEncoding module defined in 
    https://github.com/pytorch/examples/tree/master/word_language_model. 
    This reimplemetation, in contrast to the original one, caters for 
    adding sequence information in input embeddings where the batch 
    dimension comes first (``batch_first=True`). 

    Parameters
    ----------
    d_model : int
        The embedding dimension adopted by the associated Transformer. 
    dropout : float
        Dropout value. Dropout is applied over the sum of the input 
        embeddings and the positional encoding vectors. 
    max_len : int
        the max length of the incoming sequence. By default 10000. 
    """


    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()

        # Check if d_model is an integer and is even
        assert isinstance(d_model, int), "d_model must be an integer"
        assert d_model % 2 == 0, "d_model must be an even number"
        
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('pe', pe) # shape (max_len, d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Sequence of prefix event tokens or suffix event tokens fed 
            to the positional encoding module. Shape 
            (batch_size, window_size, d_model).

        Returns
        -------
        Updated sequence tensor of the same shape, with sequence 
        information injected into it, and dropout applied. 
        """
        x = x + self.pe[:x.size(1), :] # (batch_size, window_size, d_model)
        return self.dropout(x)

class SuTraN(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 cardinality_categoricals_pref, 
                 num_numericals_pref, 
                 num_prefix_encoder_layers = 3, 
                 num_decoder_layers = 2,
                 num_heads=8, 
                 d_ff = 128, 
                 dropout = 0.2, 
                 remaining_runtime_head = True, 
                 layernorm_embeds = True, 
                 outcome_bool = False,
                 activation = "relu",
                 ):
        """Initialize an instance of SuTraN. The learned 
        activity embedding weight matrix is shared between the encoder 
        and decoder. 

        Parameters
        ----------
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix and suffix, no END token is 
            included. Hence, the amount of distinct levels there is 
            equal to `num_activities`-1. 
        d_model : int
            Model dimension. Each sublayer of the encoder and decoder 
            blocks take as input a (batch_size, window_size, d_model) 
            shaped tensor, and output an updated tensor of the same 
            shape. 
        cardinality_categoricals_pref : list of int
            List of `num_categoricals` integers. Each integer entry 
            i (i = 0, ..., `num_categoricals`-1) contains the cardinality 
            of the i'th categorical feature of the encoder prefix events. 
            The order of the cardinalities should match the order in 
            which the categoricals are fed as inputs. Note that for each 
            categorical, an extra category should be included to account 
            for missing values.
        num_numericals_pref : int 
            Number of numerical features of the prefix events
        num_prefix_encoder_layers : int, optional
            The number of prefix encoder blocks stacked on top of each 
            other, by default 3.
        num_decoder_layers : int, optional
            Number of decoder blocks stacked on top of each other, 
            by default 2.
        num_heads : int, optional
            Number of attention heads for the Multi-Head Attention 
            sublayers in both the encoder and decoder blocks, by default 
            8.
        d_ff : int, optional
            The dimension of the hidden layer of the point-wise feed 
            forward sublayers in the transformer blocks , by default 128.
        dropout : float, optional
            Dropout rate during training. By default 0.2. 
        remaining_runtime_head : bool, optional
            If True, on top of the default time till next event suffix 
            prediction and the activity suffix prediction, also the 
            complete remaining runtime is predicted. By default True. 
            See Notes for further remarks 
            regarding the `remaining_runtime_head` parameter. 
        layernorm_embeds : bool, optional
            Whether or not Layer Normalization is applied over the 
            initial embeddings of the encoder and decoder. True by 
            default.
        outcome_bool : bool, optional 
            Whether or not the model should also include a prediction 
            head for binary outcome prediction. By default `False`. If 
            `outcome_bool=True`, a prediction head for predicting 
            the binary outcome given a prefix is added. This prediction 
            head, in contrast to the time till next event and activity 
            suffix predictions, will only be trained to provide a 
            prediction at the first decoding step. Note that the 
            value of `outcome_bool` should be aligned with the 
            `outcome_bool` parameter of the training and inference 
            procedure, as well as with the preprocessing pipeline that 
            produces the labels. See Notes for further remarks 
            regarding the `outcome_bool` parameter. 

        Notes
        -----
        Additional remarks regarding parameters: 

        * `remaining_runtime_head` : This parameter has become redundant, and 
        should always be set to `True`. SuTraN by default accounts for an 
        additional direct remaining runtime prediction head. 

        * `outcome_bool` : For the paper implementation, this boolean should 
        be set to `False`. For future work, already included for extending 
        the multi-task PPM setup to simultaneously predict a binary outcome 
        target for each prefix as well.  
        """
        super(SuTraN, self).__init__()

        self.num_activities = num_activities

        self.d_model = d_model

        # Cardinality categoricals encoder prefix events
        self.cardinality_categoricals_pref = cardinality_categoricals_pref
        self.num_categoricals_pref = len(self.cardinality_categoricals_pref)

        # Number of numerical features encoder prefix events 
        self.num_numericals_pref = num_numericals_pref

        self.num_prefix_encoder_layers = num_prefix_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.remaining_runtime_head = remaining_runtime_head
        self.layernorm_embeds = layernorm_embeds
        self.outcome_bool = outcome_bool
        self.activation = activation

        # Initialize positional encoding layer 
        self.positional_encoding = PositionalEncoding(d_model)

        # Initializing the categorical embeddings for the encoder inputs: 
        # Shared activity embeddings prefix and suffix! So only for the remaining ones you should do it. 
        self.embed_sz_categ_pref = [min(600, round(1.6 * n_cat**0.56)) for n_cat in self.cardinality_categoricals_pref[:-1]]
        self.activity_emb_size = min(600, round(1.6 * self.cardinality_categoricals_pref[-1]**0.56))

        # Initializing a separate embedding layer for each categorical prefix feature 
        # (Incrementing the cardinality with 1 to account for the padding idx of 0.)
        self.cat_embeds_pref = nn.ModuleList([nn.Embedding(num_embeddings=self.cardinality_categoricals_pref[i]+1, embedding_dim=self.embed_sz_categ_pref[i], padding_idx=0) for i in range(self.num_categoricals_pref-1)])
        self.act_emb = nn.Embedding(num_embeddings=num_activities-1, embedding_dim=self.activity_emb_size, padding_idx=0)


        # Dimensionality of initial encoder events after the prefix categoricals are fed to the dedicated entity embeddings and everything, including the numericals 
        # are concatenated
        self.dim_init_prefix = sum(self.embed_sz_categ_pref) + self.activity_emb_size + self.num_numericals_pref
        # Initial input embedding prefix events (encoder)
        self.input_embeddings_encoder = nn.Linear(self.dim_init_prefix, self.d_model)

        # Dimensionality of initial decoder suffix event tokens after the suffix categoricals are fed to the dedicated entity embeddings and everything, 
        # including the numericals are concatenated
        self.dim_init_suffix = self.activity_emb_size + 2

        # Initial input embedding prefix events (encoder)
        self.input_embeddings_decoder = nn.Linear(self.dim_init_suffix, self.d_model)

        # Initializing the num_prefix_encoder_layers encoder layers 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, activation) for _ in range(self.num_prefix_encoder_layers)])
        # Initializing the num_decoder_layers decoder layers (for training with teacher forcing)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, activation) for _ in range(self.num_decoder_layers)])
        # Initializing cached decoder layers for fast autoregressive inference
        self.decoder_layers_cached = nn.ModuleList([DecoderLayerCached(d_model, num_heads, d_ff, dropout, activation) for _ in range(self.num_decoder_layers)])

        # Initializing the additional activity output layer
        self.fc_out_act = nn.Linear(self.d_model, self.num_activities) # (batch_size, window_size, num_activities)

        # Initializing the additional time till next event prediction layer
        self.fc_out_ttne = nn.Linear(self.d_model, 1)

        if self.remaining_runtime_head:
            # Additional remaining runtime layers
            self.fc_out_rrt = nn.Linear(self.d_model, 1)

        if self.outcome_bool:
            # Additional (binary) outcome head 
            self.fc_out_out = nn.Linear(self.d_model, 1)
            # Sigmoid activiation function
            self.sigmoid_out = nn.Sigmoid()
        
        
        if self.layernorm_embeds:
            self.norm_enc_embeds = nn.LayerNorm(self.d_model)
            self.norm_dec_embeds = nn.LayerNorm(self.d_model)

            
        self.dropout = nn.Dropout(self.dropout)

        # Creating forward call bools to know what to output 
        self.only_rrt = (not self.outcome_bool) & self.remaining_runtime_head
        self.only_out = self.outcome_bool & (not self.remaining_runtime_head)
        self.both_not = (not self.outcome_bool) & (not self.remaining_runtime_head)
        self.both = self.outcome_bool & self.remaining_runtime_head

        # Tie weights between decoder_layers and decoder_layers_cached
        # so training updates one set of parameters and inference reads
        # them through the cached wrappers.
        self._tie_cached_decoder_weights()

    def _tie_cached_decoder_weights(self):
        """Tie the weights of cached decoder layers to the original
        decoder layers. Both share the same nn.Linear / LayerNorm
        modules, so any gradient update to the originals is
        automatically visible to the cached layers."""
        for i in range(len(self.decoder_layers)):
            orig = self.decoder_layers[i]
            cached = self.decoder_layers_cached[i]

            # Self-attention
            cached.self_attn.W_q = orig.self_attn.W_q
            cached.self_attn.W_k = orig.self_attn.W_k
            cached.self_attn.W_v = orig.self_attn.W_v
            cached.self_attn.W_o = orig.self_attn.W_o

            # Cross-attention
            cached.cross_attn.W_q = orig.cross_attn.W_q
            cached.cross_attn.W_k = orig.cross_attn.W_k
            cached.cross_attn.W_v = orig.cross_attn.W_v
            cached.cross_attn.W_o = orig.cross_attn.W_o

            # Feed-forward (shares activation too)
            cached.feed_forward = orig.feed_forward

            # Layer norms
            cached.norm1 = orig.norm1
            cached.norm2 = orig.norm2
            cached.norm3 = orig.norm3


    # window_size : number of decoding steps during inference (model.eval())
    def forward(self, 
                inputs, 
                window_size=None, 
                mean_std_ttne=None, 
                mean_std_tsp=None, 
                mean_std_tss=None):
        """Processing a batch of inputs. The activity labels of the 
        prefix events are (and should) always be located at 
        inputs[self.num_categoricals_pref-1].

        Parameters
        ----------
        inputs : list of torch.Tensor
            List of tensors containing the various components 
            of the inputs. 
        window_size : None or int, optional
            The (shared) sequence length of the prefix and suffix inputs. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_ttne : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Till Next Event (TTNE) prediction 
            targets in seconds, computed over the training set instances 
            and used to standardize the TTNE labels of the training set, 
            validation set and test set. Needed for converting 
            timestamp predictions back to seconds and vice versa, during 
            inference only. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_tsp : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Since Previous (TSP) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSP values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_tss : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Since Start (TSS) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSS values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        """
        # Tensor containing the numerical features of the prefix events. 
        num_ftrs_pref = inputs[(self.num_categoricals_pref-1)+1] # (batch_size, window_size, N)

        # Tensor containing the padding mask for the prefix events. 
        padding_mask_input = inputs[(self.num_categoricals_pref-1)+2] # (batch_size, window_size) = (B, W)

        # Just auxilary index for understandability
        idx = self.num_categoricals_pref+2

        # Tensor containing the numerical features of the suffix event tokens: 
        num_ftrs_suf = inputs[idx + 1] # (batch_size, window_size, 2)

        # Constructing categorical embeddings prefix (encoder)
        cat_emb_pref = self.cat_embeds_pref[0](inputs[0]) # (batch_size, window_size, embed_sz_categ[0])
        for i in range(1, self.num_categoricals_pref-1):
            cat_emb_help = self.cat_embeds_pref[i](inputs[i]) # (batch_size, window_size, embed_sz_categ[i])
            cat_emb_pref = torch.cat((cat_emb_pref, cat_emb_help), dim = -1) # (batch_size, window_size, sum(embed_sz_categ[:i+1]))
        act_emb_pref = self.act_emb(inputs[self.num_categoricals_pref-1])
        cat_emb_pref = torch.cat((cat_emb_pref, act_emb_pref), dim=-1)
        
        # Concatenate cat_emb with the numerical features to get initial vector representations prefix events. 
        x = torch.cat((cat_emb_pref, num_ftrs_pref), dim = -1) # (batch_size, window_size, sum(embed_sz_categ)+N)

        # Dropout over concatenated features: 
        x = self.dropout(x)

        # Initial embedding encoder (prefix events)
        x = self.positional_encoding(self.input_embeddings_encoder(x) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)
        if self.layernorm_embeds:
            x = self.norm_enc_embeds(x) # (batch_size, window_size, d_model)

        # Updating the prefix event embeddings with the encoder blocks 
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, padding_mask_input)

        # ---------------------------

        if self.training: # Teacher forcing (for now)

            # Using the activity embedding layer shared with the encoder 
            cat_emb_suf = self.act_emb(inputs[idx]) # (batch_size, window_size, embed_sz_categ[0])
            
            # Concatenate cat_emb with the numerical features to get initial vector representations suffix event tokens.
            target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (batch_size, window_size, self.dim_init_suffix)
            
            # Initial embeddings decoder suffix event tokens 
            # The positional encoding module applies dropout over the result 
            target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)

            if self.layernorm_embeds:
                target_in = self.norm_dec_embeds(target_in) # (batch_size, window_size, d_model)

            # Activating the decoder
            dec_output = target_in
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

            # Next activity prediction head: 
            act_probs = self.fc_out_act(dec_output) # (batch_size, window_size, self.num_activities)

            # Time till next event prediction (ttne) head:
            ttne_pred = self.fc_out_ttne(dec_output) # (batch_size, window_size, 1)

            # if self.remaining_runtime_head:
            if self.only_rrt:
                # Complete remaining runtime prediction (rrt) head
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                return act_probs, ttne_pred, rrt_pred 
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1), (batch_size, window_size, 1)
            elif self.only_out:
                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, out_pred
            elif self.both:
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, rrt_pred, out_pred
            else: 
                return act_probs, ttne_pred
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1)

        else: # Inference mode: greedy decoding with KV-caching
            # Instead of re-embedding and re-decoding the full sequence
            # at every step, we process one token at a time and cache
            # the K/V projections. This reduces decoder complexity from
            # O(W^2 * L) to O(W * L) per sequence.

            act_inputs = inputs[idx] # (B, W)

            batch_size = act_inputs.size(0) # B

            # Output tensors
            suffix_acts_decoded = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.int64).to(device)
            suffix_ttne_preds = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.float32).to(device)

            # Initialize KV caches (one per decoder layer)
            num_layers = len(self.decoder_layers_cached)
            self_attn_caches_k = [None] * num_layers
            self_attn_caches_v = [None] * num_layers
            cross_attn_caches_k = [None] * num_layers
            cross_attn_caches_v = [None] * num_layers

            # Start with the first suffix token
            current_act = act_inputs[:, 0:1]           # (B, 1)
            current_time_ftrs = num_ftrs_suf[:, 0:1, :]  # (B, 1, 2)

            for dec_step in range(0, window_size):
                # Embed only the current token
                cat_emb_suf = self.act_emb(current_act)  # (B, 1, activity_emb_size)
                target_in = torch.cat((cat_emb_suf, current_time_ftrs), dim=-1)  # (B, 1, dim_init_suffix)

                # Project + positional encoding for this step only
                target_in = self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)
                target_in = target_in + self.positional_encoding.pe[dec_step:dec_step+1, :]
                target_in = self.positional_encoding.dropout(target_in)

                if self.layernorm_embeds:
                    target_in = self.norm_dec_embeds(target_in)

                # Pass through cached decoder layers
                dec_output = target_in
                for layer_idx, dec_layer in enumerate(self.decoder_layers_cached):
                    dec_output, new_self_k, new_self_v, cross_k, cross_v = dec_layer(
                        dec_output,
                        x,  # encoder output
                        padding_mask_input,
                        self_attn_cache_k=self_attn_caches_k[layer_idx],
                        self_attn_cache_v=self_attn_caches_v[layer_idx],
                        cross_attn_cache_k=cross_attn_caches_k[layer_idx],
                        cross_attn_cache_v=cross_attn_caches_v[layer_idx],
                    )
                    self_attn_caches_k[layer_idx] = new_self_k
                    self_attn_caches_v[layer_idx] = new_self_v
                    cross_attn_caches_k[layer_idx] = cross_k
                    cross_attn_caches_v[layer_idx] = cross_v

                # dec_output: (B, 1, d_model)

                # Activity prediction
                act_logits = self.fc_out_act(dec_output)  # (B, 1, num_activities)
                act_outputs = act_logits[:, 0, :]  # (B, num_activities)

                # TTNE prediction
                ttne_pred = self.fc_out_ttne(dec_output)  # (B, 1, 1)
                ttne_outputs = ttne_pred[:, 0, 0]  # (B,)
                suffix_ttne_preds[:, dec_step] = ttne_outputs

                # RRT / outcome only at first decoding step
                if dec_step == 0:
                    if self.remaining_runtime_head:
                        rrt_pred = self.fc_out_rrt(dec_output)[:, 0, 0]  # (B,)
                    if self.outcome_bool:
                        out_pred = self.sigmoid_out(self.fc_out_out(dec_output))[:, 0, 0]  # (B,)

                # Greedy decode — mask padding token
                act_outputs[:, 0] = -1e9
                act_selected = torch.argmax(act_outputs, dim=-1)  # (B,)
                suffix_acts_decoded[:, dec_step] = act_selected

                if dec_step < (window_size - 1):
                    # Prepare next token
                    act_suf_updates = torch.clamp(act_selected, max=self.num_activities - 2)
                    current_act = act_suf_updates.unsqueeze(1)  # (B, 1)

                    # Derive next time features from TTNE prediction
                    time_preds_seconds = ttne_outputs * mean_std_ttne[1] + mean_std_ttne[0]
                    time_preds_seconds = torch.clamp(time_preds_seconds, min=0)

                    tss_stand = current_time_ftrs[:, 0, 0]
                    tss_seconds = tss_stand * mean_std_tss[1] + mean_std_tss[0]
                    tss_seconds = torch.clamp(tss_seconds, min=0)

                    tss_seconds_new = tss_seconds + time_preds_seconds
                    tss_stand_new = (tss_seconds_new - mean_std_tss[0]) / mean_std_tss[1]
                    tsp_stand_new = (time_preds_seconds - mean_std_tsp[0]) / mean_std_tsp[1]

                    current_time_ftrs = torch.stack([tss_stand_new, tsp_stand_new], dim=-1).unsqueeze(1)  # (B, 1, 2)

            if self.only_rrt:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred
            elif self.only_out:
                return suffix_acts_decoded, suffix_ttne_preds, out_pred
            elif self.both:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred, out_pred
            else:
                return suffix_acts_decoded, suffix_ttne_preds




# ---------------------------------------


class SuTraN_no_context(nn.Module):
    def __init__(self, 
                 num_activities, 
                 d_model, 
                 num_prefix_encoder_layers = 3, 
                 num_decoder_layers = 2,
                 num_heads=8, 
                 d_ff = 128, 
                 dropout = 0.2, 
                 remaining_runtime_head = True, 
                 layernorm_embeds = True, 
                 outcome_bool = False,
                 activation = "relu",
                 ):
        """Initialize an instance of SuTraN. The learned 
        activity embedding weight matrix is shared between the encoder 
        and decoder. 

        Parameters
        ----------
        num_activities : int
            Number of distinct activities present in the event log. 
            This does include the end token and padding token 
            used for the activity labels. For the categorical activity 
            label features in the prefix and suffix, no END token is 
            included. Hence, the amount of distinct levels there is 
            equal to `num_activities`-1. 
        d_model : int
            Model dimension. Each sublayer of the encoder and decoder 
            blocks take as input a (batch_size, window_size, d_model) 
            shaped tensor, and output an updated tensor of the same 
            shape. 
        num_prefix_encoder_layers : int, optional
            The number of prefix encoder blocks stacked on top of each 
            other, by default 3.
        num_decoder_layers : int, optional
            Number of decoder blocks stacked on top of each other, 
            by default 2.
        num_heads : int, optional
            Number of attention heads for the Multi-Head Attention 
            sublayers in both the encoder and decoder blocks, by default 
            8.
        d_ff : int, optional
            The dimension of the hidden layer of the point-wise feed 
            forward sublayers in the transformer blocks , by default 128.
        dropout : float, optional
            Dropout rate during training. By default 0.2. 
        remaining_runtime_head : bool, optional
            If True, on top of the default time till next event suffix 
            prediction and the activity suffix prediction, also the 
            complete remaining runtime is predicted. By default True. 
            See Notes for further remarks 
            regarding the `remaining_runtime_head` parameter. 
        layernorm_embeds : bool, optional
            Whether or not Layer Normalization is applied over the 
            initial embeddings of the encoder and decoder. True by 
            default.
        outcome_bool : bool, optional 
            Whether or not the model should also include a prediction 
            head for binary outcome prediction. By default `False`. If 
            `outcome_bool=True`, a prediction head for predicting 
            the binary outcome given a prefix is added. This prediction 
            head, in contrast to the time till next event and activity 
            suffix predictions, will only be trained to provide a 
            prediction at the first decoding step. Note that the 
            value of `outcome_bool` should be aligned with the 
            `outcome_bool` parameter of the training and inference 
            procedure, as well as with the preprocessing pipeline that 
            produces the labels. See Notes for further remarks 
            regarding the `outcome_bool` parameter. 
        activation : str, optional
            Activation function for the feed-forward sublayers. One of 
            ``"relu"``, ``"gelu"``, or ``"silu"``. By default ``"relu"``.

        Notes
        -----
        Additional remarks regarding parameters: 

        * `remaining_runtime_head` : This parameter has become redundant, and 
        should always be set to `True`. SuTraN by default accounts for an 
        additional direct remaining runtime prediction head. 

        * `outcome_bool` : For the paper implementation, this boolean should 
        be set to `False`. For future work, already included for extending 
        the multi-task PPM setup to simultaneously predict a binary outcome 
        target for each prefix as well.  
        """
        super(SuTraN_no_context, self).__init__()

        self.num_activities = num_activities

        self.d_model = d_model
        self.num_prefix_encoder_layers = num_prefix_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.remaining_runtime_head = remaining_runtime_head
        self.layernorm_embeds = layernorm_embeds
        self.outcome_bool = outcome_bool
        self.activation = activation

        # Initialize positional encoding layer 
        self.positional_encoding = PositionalEncoding(d_model)

        # Initializing the shared categorical activity embedding for the 
        # encoder and decoder input sequences (seq of prefix event tokens 
        # and seq of suffix event tokens): 
        self.activity_emb_size = min(600, round(1.6 * (self.num_activities-2)**0.56))
        self.act_emb = nn.Embedding(num_embeddings=num_activities-1, embedding_dim=self.activity_emb_size, padding_idx=0)


        # Dimensionality of initial prefix event tokens after prefix 
        # categoricals are fed to the dedicated entity embeddings and 
        # everything, including the numericals is concatenated
        self.dim_init_prefix = self.activity_emb_size + 2

        # Initial input embedding prefix events (encoder)
        self.input_embeddings_encoder = nn.Linear(self.dim_init_prefix, self.d_model)

        # Dimensionality of initial decoder suffix event tokens after the suffix categoricals are fed to the dedicated entity embeddings and everything, 
        # including the numericals are concatenated
        self.dim_init_suffix = self.activity_emb_size + 2

        # Initial input embedding prefix events (encoder)
        self.input_embeddings_decoder = nn.Linear(self.dim_init_suffix, self.d_model)

        # Initializing the num_prefix_encoder_layers encoder layers 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, activation) for _ in range(self.num_prefix_encoder_layers)])
        # Initializing the num_decoder_layers decoder layers (for training with teacher forcing)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, activation) for _ in range(self.num_decoder_layers)])
        # Initializing cached decoder layers for fast autoregressive inference
        self.decoder_layers_cached = nn.ModuleList([DecoderLayerCached(d_model, num_heads, d_ff, dropout, activation) for _ in range(self.num_decoder_layers)])

        # Initializing the additional activity output layer
        self.fc_out_act = nn.Linear(self.d_model, self.num_activities) # (batch_size, window_size, num_activities)

        # Initializing the additional time till next event prediction layer
        self.fc_out_ttne = nn.Linear(self.d_model, 1)

        if self.remaining_runtime_head:
            # Additional remaining runtime layers
            self.fc_out_rrt = nn.Linear(self.d_model, 1)

        if self.outcome_bool:
            # Additional (binary) outcome head 
            self.fc_out_out = nn.Linear(self.d_model, 1)
            # Sigmoid activiation function
            self.sigmoid_out = nn.Sigmoid()
        
        
        if self.layernorm_embeds:
            self.norm_enc_embeds = nn.LayerNorm(self.d_model)
            self.norm_dec_embeds = nn.LayerNorm(self.d_model)

            
        self.dropout = nn.Dropout(self.dropout)

        # Creating forward call bools to know what to output 
        self.only_rrt = (not self.outcome_bool) & self.remaining_runtime_head
        self.only_out = self.outcome_bool & (not self.remaining_runtime_head)
        self.both_not = (not self.outcome_bool) & (not self.remaining_runtime_head)
        self.both = self.outcome_bool & self.remaining_runtime_head

        # Tie the cached decoder layer weights to the original decoder
        # layer weights so that training updates are automatically
        # reflected in the cached inference path.
        self._tie_cached_decoder_weights()

    def _tie_cached_decoder_weights(self):
        """Share weight tensors between ``decoder_layers`` (used during
        teacher-forced training) and ``decoder_layers_cached`` (used
        during autoregressive inference with KV-caching).

        This is called once at the end of ``__init__``. Because only
        references are stored, any gradient update to the training
        layers automatically applies to the cached layers as well.
        """
        for orig, cached in zip(self.decoder_layers, self.decoder_layers_cached):
            # ---------- self-attention ----------
            cached.self_attn.W_q = orig.self_attn.W_q
            cached.self_attn.W_k = orig.self_attn.W_k
            cached.self_attn.W_v = orig.self_attn.W_v
            cached.self_attn.W_o = orig.self_attn.W_o

            # ---------- cross-attention ----------
            cached.cross_attn.W_q = orig.cross_attn.W_q
            cached.cross_attn.W_k = orig.cross_attn.W_k
            cached.cross_attn.W_v = orig.cross_attn.W_v
            cached.cross_attn.W_o = orig.cross_attn.W_o

            # ---------- feed-forward (share entire sub-module) ----------
            cached.feed_forward = orig.feed_forward

            # ---------- layer norms ----------
            cached.norm1 = orig.norm1
            cached.norm2 = orig.norm2
            cached.norm3 = orig.norm3


    # window_size : number of decoding steps during inference (model.eval())
    def forward(self, 
                inputs, 
                window_size=None, 
                mean_std_ttne=None, 
                mean_std_tsp=None, 
                mean_std_tss=None):
        """Processing a batch of inputs. The activity labels of the 
        prefix events are (and should) always be located at 
        inputs[self.num_categoricals_pref-1].

        Parameters
        ----------
        inputs : list of torch.Tensor
            List of tensors containing the various components 
            of the inputs. 
        window_size : None or int, optional
            The (shared) sequence length of the prefix and suffix inputs. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_ttne : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Till Next Event (TTNE) prediction 
            targets in seconds, computed over the training set instances 
            and used to standardize the TTNE labels of the training set, 
            validation set and test set. Needed for converting 
            timestamp predictions back to seconds and vice versa, during 
            inference only. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_tsp : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Since Previous (TSP) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSP values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        mean_std_tss : None or list of int, optional 
            List of two integers representing the mean and standard 
            deviation of the Time Since Start (TSS) event features of 
            the suffix event tokens, in seconds computed over the 
            training set instances and used to standardize the TSS values 
            of the training set, validation set and test set. 
            Only needed during inference (`model.eval()`). The default 
            value `None` can be retained during training. 
        """
        # Tensor containing the numerical features of the prefix events. 
        num_ftrs_pref = inputs[1] # (batch_size, window_size, N)

        # Tensor containing the padding mask for the prefix events. 
        padding_mask_input = inputs[2] # (batch_size, window_size) = (B, W)

        # Tensor containing the numerical features of the suffix event tokens: 
        num_ftrs_suf = inputs[4] # (batch_size, window_size, 2)

        # # Constructing categorical embeddings prefix (encoder)
        act_emb_pref = self.act_emb(inputs[0])
        # cat_emb_pref = torch.cat((cat_emb_pref, act_emb_pref), dim=-1)
        
        # Concatenate cat_emb with the numerical features to get initial vector representations prefix events. 
        x = torch.cat((act_emb_pref, num_ftrs_pref), dim = -1) # (batch_size, window_size, sum(embed_sz_categ)+N)

        # Dropout over concatenated features: 
        x = self.dropout(x)

        # Initial embedding encoder (prefix events)
        x = self.positional_encoding(self.input_embeddings_encoder(x) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)
        if self.layernorm_embeds:
            x = self.norm_enc_embeds(x) # (batch_size, window_size, d_model)

        # Updating the prefix event embeddings with the encoder blocks 
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, padding_mask_input)

        # ---------------------------

        if self.training: # Teacher forcing (for now)


            # Using the activity embedding layer shared with the encoder 
            cat_emb_suf = self.act_emb(inputs[3]) # (batch_size, window_size, embed_sz_categ[0])
            
            # Concatenate cat_emb with the numerical features to get initial vector representations suffix event tokens.
            target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (batch_size, window_size, self.dim_init_suffix)
            
            # Initial embeddings decoder suffix event tokens 
            target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)

            if self.layernorm_embeds:
                target_in = self.norm_dec_embeds(target_in) # (batch_size, window_size, d_model)

            # Activating the decoder
            dec_output = target_in
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

            # Next activity prediction head: 
            act_probs = self.fc_out_act(dec_output) # (batch_size, window_size, self.num_activities)

            # Time till next event prediction (ttne) head:
            ttne_pred = self.fc_out_ttne(dec_output) # (batch_size, window_size, 1)

            # if self.remaining_runtime_head:
            if self.only_rrt:
                # Complete remaining runtime prediction (rrt) head
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                return act_probs, ttne_pred, rrt_pred 
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1), (batch_size, window_size, 1)
            elif self.only_out:
                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, out_pred
            elif self.both:
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, rrt_pred, out_pred
            else: 
                return act_probs, ttne_pred
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1)

        else: # Inference mode: greedy decoding with KV-caching
            # Instead of re-embedding and re-decoding the full sequence
            # at every step, we process one token at a time and cache
            # the K/V projections. This reduces decoder complexity from
            # O(W^2 * L) to O(W * L) per sequence.

            act_inputs = inputs[3] # (B, W)

            batch_size = act_inputs.size(0) # B

            # Output tensors
            suffix_acts_decoded = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.int64).to(device)
            suffix_ttne_preds = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.float32).to(device)

            # Initialize KV caches (one per decoder layer)
            num_layers = len(self.decoder_layers_cached)
            self_attn_caches_k = [None] * num_layers
            self_attn_caches_v = [None] * num_layers
            cross_attn_caches_k = [None] * num_layers
            cross_attn_caches_v = [None] * num_layers

            # Start with the first suffix token
            current_act = act_inputs[:, 0:1]           # (B, 1)
            current_time_ftrs = num_ftrs_suf[:, 0:1, :]  # (B, 1, 2)

            for dec_step in range(0, window_size):
                # Embed only the current token
                cat_emb_suf = self.act_emb(current_act)  # (B, 1, activity_emb_size)
                target_in = torch.cat((cat_emb_suf, current_time_ftrs), dim=-1)  # (B, 1, dim_init_suffix)

                # Project + positional encoding for this step only
                target_in = self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)
                target_in = target_in + self.positional_encoding.pe[dec_step:dec_step+1, :]
                target_in = self.positional_encoding.dropout(target_in)

                if self.layernorm_embeds:
                    target_in = self.norm_dec_embeds(target_in)

                # Pass through cached decoder layers
                dec_output = target_in
                for layer_idx, dec_layer in enumerate(self.decoder_layers_cached):
                    dec_output, new_self_k, new_self_v, cross_k, cross_v = dec_layer(
                        dec_output,
                        x,  # encoder output
                        padding_mask_input,
                        self_attn_cache_k=self_attn_caches_k[layer_idx],
                        self_attn_cache_v=self_attn_caches_v[layer_idx],
                        cross_attn_cache_k=cross_attn_caches_k[layer_idx],
                        cross_attn_cache_v=cross_attn_caches_v[layer_idx],
                    )
                    self_attn_caches_k[layer_idx] = new_self_k
                    self_attn_caches_v[layer_idx] = new_self_v
                    cross_attn_caches_k[layer_idx] = cross_k
                    cross_attn_caches_v[layer_idx] = cross_v

                # dec_output: (B, 1, d_model)

                # Activity prediction
                act_logits = self.fc_out_act(dec_output)  # (B, 1, num_activities)
                act_outputs = act_logits[:, 0, :]  # (B, num_activities)

                # TTNE prediction
                ttne_pred = self.fc_out_ttne(dec_output)  # (B, 1, 1)
                ttne_outputs = ttne_pred[:, 0, 0]  # (B,)
                suffix_ttne_preds[:, dec_step] = ttne_outputs

                # RRT / outcome only at first decoding step
                if dec_step == 0:
                    if self.remaining_runtime_head:
                        rrt_pred = self.fc_out_rrt(dec_output)[:, 0, 0]  # (B,)
                    if self.outcome_bool:
                        out_pred = self.sigmoid_out(self.fc_out_out(dec_output))[:, 0, 0]  # (B,)

                # Greedy decode — mask padding token
                act_outputs[:, 0] = -1e9
                act_selected = torch.argmax(act_outputs, dim=-1)  # (B,)
                suffix_acts_decoded[:, dec_step] = act_selected

                if dec_step < (window_size - 1):
                    # Prepare next token
                    act_suf_updates = torch.clamp(act_selected, max=self.num_activities - 2)
                    current_act = act_suf_updates.unsqueeze(1)  # (B, 1)

                    # Derive next time features from TTNE prediction
                    time_preds_seconds = ttne_outputs * mean_std_ttne[1] + mean_std_ttne[0]
                    time_preds_seconds = torch.clamp(time_preds_seconds, min=0)

                    tss_stand = current_time_ftrs[:, 0, 0]
                    tss_seconds = tss_stand * mean_std_tss[1] + mean_std_tss[0]
                    tss_seconds = torch.clamp(tss_seconds, min=0)

                    tss_seconds_new = tss_seconds + time_preds_seconds
                    tss_stand_new = (tss_seconds_new - mean_std_tss[0]) / mean_std_tss[1]
                    tsp_stand_new = (time_preds_seconds - mean_std_tsp[0]) / mean_std_tsp[1]

                    current_time_ftrs = torch.stack([tss_stand_new, tsp_stand_new], dim=-1).unsqueeze(1)  # (B, 1, 2)

            if self.only_rrt:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred
            elif self.only_out:
                return suffix_acts_decoded, suffix_ttne_preds, out_pred
            elif self.both:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred, out_pred
            else:
                return suffix_acts_decoded, suffix_ttne_preds