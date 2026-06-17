import os
import time
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    """
    Decode

    Description:
        Decode class for converting CTC network probability outputs into readable gloss sequences.
        Supports two decoding modes:
            - 'max'  : greedy decoding via argmax per frame (fast, lower accuracy).
            - 'beam' : beam search via CTCBeamDecoder (more accurate, slower).

        Used after the CTC classification stage to produce gloss sequence predictions
        from the BiLSTM-CTC output.

    Input (constructor):
        - gloss_dict  (dict) : dictionary with two subkeys:
              'id2gloss'  → {str_id: {'gloss': gloss_name, ...}}
              'gloss2id'  → {gloss_name: {'index': int_id, ...}}
        - num_classes (int)  : number of gloss classes including the blank token.
        - search_mode (str)  : decoding mode, 'max' or other (beam search).
        - blank_id    (int)  : CTC blank token index (default 0).

    Process:
        - Build two-way lookup tables: id→gloss (i2g_dict) and gloss→id (g2i_dict).
        - Initialize CTCBeamDecoder with a synthetic Unicode character vocabulary
          (not actual characters, just index placeholders for the decoder).

    Output (public attributes):
        - self.i2g_dict    (dict int→str) : mapping integer indices to gloss names.
        - self.g2i_dict    (dict str→int) : mapping gloss names to integer indices.
        - self.ctc_decoder               : ready-to-use CTCBeamDecoder instance.
    """

    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        # build mapping id (int) → gloss name from subkey 'id2gloss'
        # keys in gloss_dict are strings, converted to int to be dictionary keys
        self.i2g_dict = {int(k): v['gloss'] for k, v in gloss_dict['id2gloss'].items()}

        # build mapping gloss name → id (int) from subkey 'gloss2id'
        # used when needing to convert gloss predictions back to indices
        self.g2i_dict = {k: int(v['index']) for k, v in gloss_dict['gloss2id'].items()}

        # store number of classes (including blank) for decoder initialization
        self.num_classes = num_classes

        # store search mode: 'max' for greedy, others for beam search
        self.search_mode = search_mode

        # store CTC blank token index; default 0 follows PyTorch CTC convention
        self.blank_id = blank_id

        # create synthetic vocab: num_classes Unicode characters starting from U+4E20
        # CTCBeamDecoder needs a list of characters as vocab, but values don't matter
        # because we re-decode via i2g_dict — this is just a placeholder
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]

        # initialize CTCBeamDecoder with the synthetic vocab above
        # beam_width=10: keep 10 best hypotheses at each step
        # blank_id: position of blank token in vocab
        # num_processes=10: number of parallel threads to speed up decoding
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(
            vocab,
            beam_width=10,
            blank_id=blank_id,
            num_processes=10
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        """
        Entry point decoding. Receives network output and delegates
        to MaxDecode or BeamSearch according to the configured search_mode.
        Handles tensor permutation if format is not batch-first.

        Args:
            - nn_output  (Tensor, B×T×N atau T×B×N):
                network output logits/probabilities.
            - vid_lgt    (Tensor, B):
                valid length (frame count) for each sample in the batch.
            - batch_first(bool, default True):
                True if dim 0 is batch.
                If False (format T×B×N), tensor is permuted to B×T×N first.
            - probs      (bool, default False):
                True if nn_output already contains probabilities (post-softmax);
                False if still logits (softmax will be applied internally).

        Returns:
            - ret_list (list of list of tuple): decode result per sample in the batch.
              Each element is a list of (gloss_name, position) pairs.
        """
        if not batch_first:
            # permute from (T, B, N) to (B, T, N) format required by decoder
            nn_output = nn_output.permute(1, 0, 2)

        if self.search_mode == "max":
            # use greedy decoding: fast but not optimal
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            # use beam search: more accurate, suitable for final evaluation
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        """
        Performs CTC beam search decoding using CTCBeamDecoder.
        Maintains beam_width best hypotheses at each time step
        and selects the hypothesis with the highest score as output.

        Input:
            - nn_output (Tensor, B×T×N): network output in batch-first format.
              Must be permuted before calling.
            - vid_lgt   (Tensor, B)    : valid length for each sequence in the batch.
            - probs     (bool)         : True if already probabilities (post-softmax);
              False if still logits (will be softmaxed inside this function).

        Process:
            1. If not yet probabilities: apply softmax on dim -1 (per-frame per-class)
               then move to CPU (CTCBeamDecoder only runs on CPU).
            2. Move vid_lgt to CPU.
            3. Run CTCBeamDecoder.decode → beam_result, beam_scores,
               timesteps, out_seq_len.
            4. For each sample in the batch:
               a. Take the best hypothesis (beam index 0).
               b. Truncate according to valid length out_seq_len[batch_idx][0].
               c. Remove consecutive duplicates via groupby (CTC collapse).
               d. Convert gloss indices to gloss names via i2g_dict.

        Output:
            - ret_list (list of list of tuple): one list per sample in the batch.
              Each tuple: (gloss_name: str, position: int).
              Position is the sequential index in the decode result (not frame index).
        """
        if not probs:
            # apply softmax to make output a valid probability distribution
            # move to CPU because CTCBeamDecoder does not support GPU tensors
            nn_output = nn_output.softmax(-1).cpu()

        # move sequence lengths to CPU for consistency with nn_output
        vid_lgt = vid_lgt.cpu()

        # run beam search decoding
        # beam_result : (B, N_beams, T)  — gloss indices per hypothesis per frame
        # beam_scores : (B, N_beams)     — log-prob of each hypothesis (smaller is better)
        # timesteps   : (B, N_beams)     — frame position of each token
        # out_seq_len : (B, N_beams)     — valid length of each hypothesis
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(
            nn_output, vid_lgt
        )

        # list to collect decode results for each sample
        ret_list = []
        for batch_idx in range(len(nn_output)):
            # take best hypothesis (beam index 0) and truncate to valid length
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]

            if len(first_result) != 0:
                # remove consecutive duplicate tokens using groupby
                # (CTC collapse: "A A B B A" → "A B A")
                # x[0] takes the first unique value from each group
                first_result = torch.stack([x[0] for x in groupby(first_result)])

            # convert integer indices to gloss names and create list of (gloss, position) tuples
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(first_result)
            ])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        """
        Performs greedy CTC decoding by taking argmax per frame,
        then applies CTC collapsing rules: remove consecutive duplicates
        and remove blank tokens.

        Input:
            - nn_output (Tensor, B×T×N): network output logits (pre-softmax).
              Softmax is not required because argmax is not affected by monotonic
              transformation — maximum position is the same before and after softmax.
            - vid_lgt   (Tensor, B)    : valid length for each sequence in the batch.

        Process:
            1. Take argmax on dim 2 (per frame) → index_list (B, T).
            2. For each sample in the batch:
               a. Truncate sequence to valid length vid_lgt[batch_idx].
               b. Remove consecutive duplicates via groupby (CTC collapse step 1).
               c. Filter out blank tokens (CTC collapse step 2).
               d. Jika masih ada token tersisa, hapus duplikat sekali lagi
                  after filtering (second groupby).
               e. Convert indices to gloss names via i2g_dict.

        Output:
            - ret_list (list of list of tuple): one list per sample.
              Each tuple: (gloss_name: str, position: int).
        """
        # take the index of the class with the highest probability for each frame
        # axis=2 because the format is (B, T, N): N is the class dimension
        index_list = torch.argmax(nn_output, axis=2)

        # get batch size and maximum sequence length
        batchsize, lgt = index_list.shape

        # list to collect decode results for each sample
        ret_list = []
        for batch_idx in range(batchsize):
            # truncate sequence to valid length for this sample (remove padding)
            # then remove consecutive duplicates: "A A B B A" → "A B A" (CTC step 1)
            group_result = [
                x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])
            ]

            # remove all blank tokens from the collapsed result (CTC collapse step 2)
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]

            if len(filtered) > 0:
                # stack list of tensors into a single tensor for the next operation
                max_result = torch.stack(filtered)
                # second groupby: after removing blanks, new duplicates may appear
                # that were previously separated by blanks, e.g., "A blank A" →
                # after removing blanks becomes "A A" → collapse again → "A"
                max_result = [x[0] for x in groupby(max_result)]
            else:
                # no tokens other than blank → empty result
                max_result = filtered

            # convert integer indices to gloss names and create list of (gloss, position) tuples
            ret_list.append([
                (self.i2g_dict[int(gloss_id)], idx)
                for idx, gloss_id in enumerate(max_result)
            ])
        # Return list of list of tuple
        return ret_list