import os
# import pdb
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        """
        Prepare containers for device information.
        """
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        """
        Set the GPU used by the program from the device configuration string.
        
        Parameters
        ----------
        device : str
            String containing the GPU ID, e.g., '0' or '0,1'.
        """
        device = str(device)
        if device != 'None':
            self.gpu_list = [i for i in range(len(device.split(',')))]
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            output_device = self.gpu_list[0]
            self.occupy_gpu(self.gpu_list)
        self.output_device = output_device if len(self.gpu_list) > 0 else "cpu"

    def model_to_device(self, model):
        """
        Move the model to the main device and wrap it with DataParallel if necessary.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be moved.

        Returns
        -------
        torch.nn.Module
            The model moved to the device.
        """
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        """
        Move tensor data to the target device by adjusting its data type.

        Parameters
        ----------
        data : torch.Tensor
            The tensor data to be moved.

        Returns
        -------
        torch.Tensor
            The tensor data moved to the device.
        """
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))
    
    def dict_data_to_device(self, data_dict):
        """
        Move the contents of the data dictionary to the target device selectively.

        Parameters
        ----------
        data_dict : dict
            The data dictionary to be moved.

        Returns
        -------
        dict
            The data dictionary with contents moved to the device.
        """
        cuda_dict = {}
        for k, v in data_dict.items():
            if 'origin' in k or 'datasets' in k:
                cuda_dict[k] = v
            else:
                cuda_dict[k] = self.data_to_device(v)
        return cuda_dict

    def criterion_to_device(self, loss):
        """
        Move the loss or criterion object to the target device.

        Parameters
        ----------
        loss : torch.nn.Module
            The loss or criterion object to be moved.

        Returns
        -------
        torch.nn.Module
            The loss or criterion object moved to the device.
        """
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
        Make the GPU appear active in nvidia-smi by allocating a small tensor.

        Parameters
        ----------
        gpus : list or int, optional
            List of GPUs or a single GPU index to occupy.
        """
        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)
