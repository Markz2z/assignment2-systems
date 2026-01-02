import torch
import triton
import triton.language as tl
from einops import rearrange, einsum, reduce
import math

MAX_TILE_SIZE = 256
NUM_TILES = 32
MIN_TILE_SIZE = 16
verbose = False

def cdiv(a, b):
    return (a + b - 1) // b

class TorchAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        D_q, N_q, batch_dims = Q.shape[-1], Q.shape[-2], Q.shape[:-2]
        D_k, N_k = K.shape[-1], K.shape[-2]
        D_v, N_v = V.shape[-1], V.shape[-2]
        device = Q.device
        dtype = Q.dtype
        # sanity check
        assert N_k == N_v, "key/value sequence num should equal"
        assert D_k == D_v, "key/value dim should equal"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "expected contiguous tensors"

        # merge batch dims into 1, may be [B,H,xx]->[B,xx]
        Q = rearrange(Q, "... n_q d -> (...) n_q d")
        K = rearrange(K, "... n_k d -> (...) n_k d")
        V = rearrange(V, "... n_v d -> (...) n_v d")

        combined_batch_dims = Q.shape[0]

        # compute tile sizes
        TILE_SIZE = min(triton.next_power_of_2(MAX_TILE_SIZE // NUM_TILES), MAX_TILE_SIZE)
        ctx.Q_TILE_SIZE = max(TILE_SIZE, MIN_TILE_SIZE) # B_q
        ctx.K_TILE_SIZE = ctx.Q_TILE_SIZE # B_k

        # record partial logsumexp for every Q
        logsumexp = torch.empty((combined_batch_dims, N_q), device = device, dtype = dtype)
        # allocate output
        O = torch.empty((combined_batch_dims, N_q, D_v), device = device, dtype = dtype)

        for query_i in range(cdiv(N_q, ctx.Q_TILE_SIZE)):
            O_i = torch.zeros(combined_batch_dims, ctx.Q_TILE_SIZE, D_q, device=device, dtype=dtype)
            # [batch, ctx.Q_TILE_SIZE] -> partial expsum
            l_i = torch.zeros(combined_batch_dims, ctx.Q_TILE_SIZE, device=device, dtype=dtype)
            # [batch, ctx.Q_TILE_SIZE], max row_max
            m_i = torch.full((combined_batch_dims, ctx.Q_TILE_SIZE), 1e-6, device=device, dtype=dtype)

            q_idx = query_i * ctx.Q_TILE_SIZE
            # slice q for CTA
            Q_block = Q[:, q_idx:q_idx+ctx.Q_TILE_SIZE, :]

            for k_j in range(cdiv(N_k, ctx.K_TILE_SIZE)):
                # get K and V blocks
                k_idx = k_j * ctx.K_TILE_SIZE
                K_block = K[:, k_idx:k_idx+ctx.K_TILE_SIZE, :]
                V_block = V[:, k_idx:k_idx+ctx.K_TILE_SIZE, :]
                # QxKt, S_ij: [batch, ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE]
                S_ij = einsum(Q_block, K_block, 'b i j, b k j -> b i k') / math.sqrt(D_k)
                if verbose:
                    print(f'S_ij:{S_ij}')
                # row_max: [batch, ctx.Q_TILE_SIZE]
                row_max = torch.max(S_ij, dim = -1).values
                m_i_new = torch.maximum(row_max, m_i)
                if verbose:
                    print(f'row_max shape:{row_max.shape}, m_i.shape:{m_i.shape}, m_i_new.shape:{m_i_new.shape}')
                # rm max, P_ij: [batch, ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE]
                P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))
                if verbose:
                    print(f"P_ij", P_ij)
                # accumulate row exp sum(already rm current max)
                l_i_new = torch.exp(m_i - m_i_new) * l_i + reduce(P_ij, 'b q k -> b q', 'sum')

                # [batch, ctx.Q_TILE_SIZE]
                diag_el = torch.exp(m_i - m_i_new)
                # [batch, ctx.Q_TILE_SIZE, D_v]
                diag = diag_el.unsqueeze(-1) * O_i
                stride_pv = einsum(P_ij, V_block, 'b q k, b k d -> b q d')
                # [batch, ctx.Q_TILE_SIZE, D_v]
                O_i_new = diag + stride_pv
                if verbose:
                    print(f'O_i_new:{O_i_new}')
                # update l_i, m_i, O_i
                l_i = l_i_new
                m_i = m_i_new
                O_i = O_i_new
            L_i = m_i + torch.log(l_i)
            if verbose:
                print(f'L_i:{L_i}')
            # [batch, ctx.Q_TILE_SIZE] (rm max + expsum)
            diag_elements = 1 / l_i
            # [ctx.Q_TILE_SIZE, ctx.Q_TILE_SIZE]
            identity = torch.eye(l_i.shape[-1], device=device, dtype=dtype)
            diag = diag_elements.unsqueeze(-1) * identity.unsqueeze(0)
            O_i_final = einsum(diag, O_i, 'b i q, b q d -> b i d')
            if verbose:
                print(f'O_i_final:{O_i_final}')
            # write output
            O[..., q_idx:q_idx + ctx.Q_TILE_SIZE,:] = O_i_final
            logsumexp[..., q_idx:q_idx + ctx.Q_TILE_SIZE] = L_i
        # save for backward
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, logsumexp)

        # reshape output
        O = O.reshape(*batch_dims, N_q, D_v)
        logsumexp = logsumexp.reshape(*batch_dims, N_q)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors                                                      
                                                                                               
        # reshape dO                                                                           
        dO = dO.reshape(-1, dO.shape[-2], dO.shape[-1])                                        
                                                                                               
        D_q, N_q, batch_dims = Q.shape[-1], Q.shape[-2], Q.shape[:-2]                          
        D_k, N_k = K.shape[-1], K.shape[-2]                                                    
        D_v, N_v = V.shape[-1], V.shape[-2]                                                    
        device = Q.device                                                                      
        dtype = Q.dtype                                                                        
                                                                                               
        scale = math.sqrt(D_k)                                                                 
        D = O * dO # element-wise product, b x N_q x D_v                                       
        D = reduce(D, '... q d -> ... q', 'sum') # rowsum of D                                 
        S = einsum(Q, K, '... q d, ... k d -> ... q k') / scale                                
                                                                                               
        if ctx.is_causal:                                                                      
            # just do a torch mask for the backward pass                                       
            mask = torch.tril(torch.ones(N_q, N_k, device = device, dtype = dtype))            
            S = S.masked_fill(mask == 0, -float('inf'))                                        
                                                                                               
        P_ij = torch.exp(S - L.unsqueeze(-1))                                                  
        dV = einsum(P_ij, dO, '... q k, ... q d -> ... k d') # N_k = N_v                       
        dP = einsum(dO, V, '... q d, ... v d -> ... q v')                                      
        dS_ij = P_ij * (dP - D.unsqueeze(-1))                                                  
        dQ = einsum(dS_ij, K, '... q k, ... k d -> ... q d') / scale                           
        dK = einsum(dS_ij, Q, '... q k, ... q d -> ... k d') / scale                           
                                                                                               
        # return None corresponding to "causal"                                                
        return dQ, dK, dV, None                                                                

