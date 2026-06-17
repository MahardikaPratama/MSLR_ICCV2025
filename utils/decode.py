import os
import time
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    """
    Decode CTC network probability outputs into gloss sequences.

    Parameters
    ----------
    gloss_dict : dict
        Dictionary with 'id2gloss' and 'gloss2id' mappings.
    num_classes : int
        Number of gloss classes including blank token.
    search_mode : str
        Decoding mode ('max' or 'beam').
    blank_id : int, optional
        CTC blank token index. Default is 0.
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
        Entry point for decoding network output using MaxDecode or BeamSearch.

        Parameters
        ----------
        nn_output : Tensor
            Network output logits or probabilities, shape (B, T, N) or (T, B, N).
        vid_lgt : Tensor
            Valid frame counts for each sample, shape (B,).
        batch_first : bool, optional
            Whether batch is the first dimension. Default is True.
        probs : bool, optional
            True if nn_output contains probabilities. Default is False.

        Returns
        -------
        list of list of tuple
            Decode results per sample, containing (gloss_name, position) pairs.
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
        Perform CTC beam search decoding using CTCBeamDecoder.

        Parameters
        ----------
        nn_output : Tensor
            Network output, shape (B, T, N).
        vid_lgt : Tensor
            Valid sequence lengths, shape (B,).
        probs : bool, optional
            True if nn_output is probabilities. Default is False.

        Returns
        -------
        list of list of tuple
            Decode results per sample, containing (gloss_name, position) pairs.
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
        Perform greedy CTC decoding by taking argmax per frame.

        Parameters
        ----------
        nn_output : Tensor
            Network output logits, shape (B, T, N).
        vid_lgt : Tensor
            Valid sequence lengths, shape (B,).

        Returns
        -------
        list of list of tuple
            Decode results per sample, containing (gloss_name, position) pairs.
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