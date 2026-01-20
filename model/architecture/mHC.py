import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable
from einops import rearrange, repeat



class MHC(nn.Module):
    """
    Docstring for MHC
    """
    def __init__(self, ):
        super().__init__()
        pass

    def width_connection(self, residuals):
        '''  '''
        pass
    
    def depth_connection(self, branch_output, residuals, *, beta):
        '''  '''
        pass

    def decorate_branch(self, branch: Callable):
        '''  '''
        assert not exists(self.branch), 'branch was already warpped on init'

        def forward_and_residual(residual, *args, **kwargs):
            # perform mHC forward
            branch_input, add_residual = self.forward(residual)
            # perform branch forward
            branch_output = branch(branch_input, *args, **kwargs)
            # add residual connection to branch
            return add_residual(branch_output)

        return forward_and_residual
    
    def forward(self, residuals, *branch_args, **branch_kwargs):
        '''  '''
        pass


class StreamEmbed(nn.Module):
    """  """
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False,
        expand_to_streams = False
    ):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):
        # 
        if self.expand_to_streams:
            residuals = repeat(residuals, 'b ... -> (b s) ...', s = self.num_streams)

        # 
        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        residuals = residuals + self.stream_embed

        # 
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... s d -> (b s) d ...', s = self.num_streams)
        else:
            residuals = rearrange(residuals, 'b ... s d -> (b s) ... d', s = self.num_streams)

        return residuals


