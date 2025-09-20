from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.distributed as dist
import math
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

def calc_logits(embeddings, kernel):
    """ calculate original logits
    """
    embeddings = l2_norm(embeddings, axis=1)
    kernel_norm = l2_norm(kernel, axis=0)
    cos_theta = torch.mm(embeddings, kernel_norm)
    cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
    with torch.no_grad():
        origin_cos = cos_theta.clone()
    return cos_theta, origin_cos


@torch.no_grad()
def all_gather_tensor(input_tensor):
    """ allgather tensor (difference size in 0-dim) from all workers
    """
    world_size = dist.get_world_size()

    tensor_size = torch.tensor([input_tensor.shape[0]], dtype=torch.int64).cuda()
    tensor_size_list = [torch.ones_like(tensor_size) for _ in range(world_size)]
    dist.all_gather(tensor_list=tensor_size_list, tensor=tensor_size, async_op=False)
    max_size = torch.cat(tensor_size_list, dim=0).max()

    padded = torch.empty(max_size.item(), *input_tensor.shape[1:], dtype=input_tensor.dtype).cuda()
    padded[:input_tensor.shape[0]] = input_tensor
    padded_list = [torch.ones_like(padded) for _ in range(world_size)]
    dist.all_gather(tensor_list=padded_list, tensor=padded, async_op=False)

    slices = []
    for ts, t in zip(tensor_size_list, padded_list):
        slices.append(t[:ts.item()])
    return torch.cat(slices, dim=0)


def calc_top1_acc(original_logits, label,ddp=False):
    """
    Compute the top1 accuracy during training
    :param original_logits: logits w/o margin, [bs, C]
    :param label: labels [bs]
    :return: acc in all gpus
    """
    assert (original_logits.size()[0] == label.size()[0])

    with torch.no_grad():
        _, max_index = torch.max(original_logits, dim=1, keepdim=False)  # local max logit
        count = (max_index == label).sum()
        if ddp:
            dist.all_reduce(count, dist.ReduceOp.SUM)

            return count.item() / (original_logits.size()[0] * dist.get_world_size())
        else:
            return count.item() / (original_logits.size()[0])

def l2_norm(input, axis=1):
    norm = torch.norm(input, p=2, dim=axis, keepdim=True)
    output = torch.div(input, norm)
    return output


class FC_ddp2(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 explicit_kernel,
                 missing_ids,
                 scale=64.0,
                 margin=0.4,
                 mode='cosface',
                 use_cifp=False,
                 reduction='mean',
                 ddp=False,
                 pretrained_kernel=None,
                 id_count_dict={}):
        """ Args:
            in_features: size of each input features
            out_features: size of each output features
            scale: norm of input feature
            margin: margin
        """
        super(FC_ddp2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features  # num of classes
        self.missing_ids = missing_ids
        self.scale = scale
        self.margin = margin
        self.mode = mode
        self.use_cifp = use_cifp
        """
        kernel is a learnable parameter matrix: each column represents an implicit feature of a face ID (used in IIE loss: obtaining y_i)
        The dimension is [512, 1000], indicating:
                512: The same as the implicit feature dimension
                1000: The number of supported face IDs
        """
        # # Parameter will be automatically registered to the model parameter list
        # ## kernel from the first epoch to the last epoch, is always updating the same matrix
        # self.kernel = Parameter(torch.Tensor(in_features, out_features))
        # nn.init.normal_(self.kernel, std=0.01)

        # Initialize according to whether there is a pre-trained kernel
        if pretrained_kernel is not None:
            # Ensure dimension matching
            assert pretrained_kernel.shape == (in_features, out_features), \
                f"Pretrained kernel shape {pretrained_kernel.shape} doesn't match expected shape ({in_features}, {out_features})"
            # Convert pre-trained kernel to Parameter
            self.kernel = Parameter(pretrained_kernel)
            logger.info("Initialized kernel from pretrained weights")
        else:
            self.kernel = Parameter(torch.Tensor(in_features, out_features))
            nn.init.normal_(self.kernel, std=0.01)
            logger.info("Initialized kernel randomly")

         
        # Add a kernel that does not require gradient to store the updated implicit feature
        self.register_buffer('updated_kernel', torch.zeros_like(self.kernel))
        self.kernel_updated = False  # Mark whether it has been updated


        # Create a new kernel, one sub-kernel for each ID
        # Get the number of keys in the dictionary
        self.id_count_dict = id_count_dict
        # Create a dictionary to store the sub-kernel for each ID
        self.new_kernel = {}
        # Traverse the dictionary, create and initialize each sub-kernel
        for key, value in self.id_count_dict.items():
            shape = (512, value)
            sub_kernel = torch.Tensor(*shape)       # Do not use nn.Parameter, to avoid modifying the kernel with backpropagation
            nn.init.normal_(sub_kernel, std=0.01)
            self.new_kernel[key] = sub_kernel

        self.ddp = ddp
        self.criteria = torch.nn.CrossEntropyLoss(reduction=reduction)

    def initialize_kernel_with_perturbation(self, out_features, explicit_kernel):
        """Initialize the implicit feature library based on the explicit feature library, add small perturbation, skip the missing ID"""
        # Copy the explicit feature library
        implicit_kernel = explicit_kernel.clone()
        
        # Add small Gaussian noise perturbation
        perturbation_std = 0.1  # Adjustable perturbation strength
        
        # Process each column (each ID) separately
        for i in range(out_features):
            if i in self.missing_ids:
                # For missing IDs, initialize with random unit vector
                random_vector = torch.randn(self.in_features, device=explicit_kernel.device)
                implicit_kernel[:, i] = l2_norm(random_vector.unsqueeze(0)).squeeze()
                logger.info(f"Initialized missing ID {i} with random unit vector")
                continue
                
            # Add perturbation to non-missing IDs
            original_col = explicit_kernel[:, i]
            noise = torch.randn_like(original_col) * perturbation_std
            perturbed_col = original_col + noise
            
            # Normalize
            perturbed_col = l2_norm(perturbed_col.unsqueeze(0)).squeeze()
            
            # Check similarity
            similarity = torch.dot(original_col, perturbed_col)
            if similarity < 0.8:  # If the similarity is too low, reduce the perturbation
                perturbed_col = 0.9 * original_col + 0.1 * perturbed_col
                perturbed_col = l2_norm(perturbed_col.unsqueeze(0)).squeeze()
            
            implicit_kernel[:, i] = perturbed_col
        
        # Finally verify
        assert not torch.isnan(implicit_kernel).any(), "NaN values in initialized kernel"
        logger.info("Successfully initialized kernel with perturbation")
        
        return Parameter(implicit_kernel)

    def apply_margin(self, target_cos_theta, label, target_index, embeddings, explicit_kernel):
        """
        Dynamically calculate margin
        Args:
            target_cos_theta: The cosine similarity between the implicit feature of each sample in the current batch and the implicit feature of the corresponding ID [bs, 1]
            label: The real/fake label of the image [batch_size]
            target_index: The target face id [batch_size]
            embeddings: The implicit feature [batch_size, 512]
            explicit_kernel: The explicit feature library [512, 1000]
        """
        assert self.mode in ['cosface', 'arcface'], 'Please check the mode'
        
        if self.mode == 'arcface':
            # The calculation of arcface remains unchanged...
            cos_m = math.cos(self.margin)
            sin_m = math.sin(self.margin)
            theta = math.cos(math.pi - self.margin)
            sinmm = math.sin(math.pi - self.margin) * self.margin
            sin_theta = torch.sqrt(1.0 - torch.pow(target_cos_theta, 2))
            cos_theta_m = target_cos_theta * cos_m - sin_theta * sin_m
            target_cos_theta_m = torch.where(
                target_cos_theta > theta, cos_theta_m, target_cos_theta - sinmm)
        elif self.mode == 'cosface':
            label = label.view(-1)
            fake_mask = (label == 1)
            real_mask = (label == 0)
            
            if real_mask.any():
                # 1. Get the features of real samples
                real_implicit_feats = embeddings[real_mask]  # [num_real, 512]
                real_target_indices = target_index[real_mask]  # [num_real]
                real_explicit_feats = explicit_kernel[:, real_target_indices].t()  # [num_real, 512]
                
                # 2. Normalize the features with L2
                real_implicit_feats = F.normalize(real_implicit_feats, p=2, dim=1)
                real_explicit_feats = F.normalize(real_explicit_feats, p=2, dim=1)
                
                # 3. Calculate the mean of cosine similarity
                real_cos_theta = torch.sum(real_implicit_feats * real_explicit_feats, dim=1)
                
                # 4. Ensure the cosine similarity is within the range of [-1, 1]
                real_cos_theta = torch.clamp(real_cos_theta, -1.0, 1.0)
                real_cos_theta = real_cos_theta.mean()
                
                # 5. Calculate the dynamic margin, and ensure it is within a reasonable range
                dynamic_margin = 0.5 * real_cos_theta
                dynamic_margin = torch.clamp(dynamic_margin, min=-0.5, max=0.5)
            else:
                dynamic_margin = torch.tensor(0.4).to(target_cos_theta.device)
                
            # Calculate the results of real and fake after adding margin
            target_cos_theta[real_mask] = target_cos_theta[real_mask] - self.margin
            target_cos_theta[fake_mask] = target_cos_theta[fake_mask] - dynamic_margin

        return target_cos_theta

    def custom_loss(self, output, target_index, scale, explicit_kernel):
        """Calculate IIE loss
        Args:
            output: [batch_size, num_classes] - The scaled similarity matrix
            target_index: [batch_size] - The id of the corresponding target face
            scale: The scaling factor
            explicit_kernel: The explicit feature library [512, 1000]
        """
        nan_mask = torch.isnan(output)
        if nan_mask.any():
            nan_positions = torch.where(nan_mask)
            print(f"Warning: NaN values found in output! Location (row, column): {list(zip(nan_positions[0].tolist(), nan_positions[1].tolist()))}")
        output = torch.abs(output)
        exp_output = torch.exp(output)  # [bs, num_classes]
        sum_exp = torch.sum(exp_output, dim=1)  # [bs]
        exp_correct = exp_output[range(output.size(0)), target_index]  # [batch_size]
        probs = exp_correct / sum_exp  # [batch_size]
        loss_im_diff = -torch.log(probs).mean()
        loss_angle_loss = self.compute_bank_loss(explicit_kernel)
        
        return loss_im_diff + loss_angle_loss

    def forward(self, embeddings, target_index, label, scale, explicit_kernel, return_logits=False):
        """
        :param embeddings: The implicit feature [batch_size, embedding_size]
        :param target_index: The id of the corresponding target face [batch_size]
        :param label: The real/fake label of the image [batch_size]
        :param scale: The scaling factor
        :param explicit_kernel: The explicit feature library [512, 1000]
        :param return_logits: Whether to return logits
        :return:
        loss: The calculated local loss, w/wo CIFP
        acc: The local accuracy
        output: The local logits, with margin, with gradient, scaled, [bs, C].
        """
        sample_num = embeddings.size(0)

        if not self.use_cifp:
            cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
            target_cos_theta = cos_theta[torch.arange(0, sample_num), target_index].view(-1, 1)   
            target_cos_theta_m = self.apply_margin(target_cos_theta, label, target_index, embeddings, explicit_kernel)
            cos_theta.scatter_(1, target_index.view(-1, 1).long(), target_cos_theta_m)
        else:
            cos_theta, origin_cos = calc_logits(embeddings, self.kernel)
            cos_theta_, _ = calc_logits(embeddings, self.kernel.detach())

            mask = torch.zeros_like(cos_theta)  # [bs，C]
            mask.scatter_(1, target_index.view(-1, 1).long(), 1.0)  # one-hot label / gt mask

            tmp_cos_theta = cos_theta - 2 * mask
            tmp_cos_theta_ = cos_theta_ - 2 * mask

            target_cos_theta = cos_theta[torch.arange(0, sample_num), target_index].view(-1, 1)
            target_cos_theta_ = cos_theta_[torch.arange(0, sample_num), target_index].view(-1, 1)

            target_cos_theta_m = self.apply_margin(target_cos_theta, label)

            far = 1 / (self.out_features - 1)  # ru+ value
            # far = 1e-5

            topk_mask = torch.greater(tmp_cos_theta, target_cos_theta)
            topk_sum = torch.sum(topk_mask.to(torch.int32))
            if self.ddp:
                dist.all_reduce(topk_sum)
            far_rank = math.ceil(far * (sample_num * (self.out_features - 1) * dist.get_world_size() - topk_sum))
            cos_theta_neg_topk = torch.topk((tmp_cos_theta - 2 * topk_mask.to(torch.float32)).flatten(),
                                            k=far_rank)[0]  # [far_rank]
            cos_theta_neg_topk = all_gather_tensor(cos_theta_neg_topk.contiguous())  # top k across all gpus
            cos_theta_neg_th = torch.topk(cos_theta_neg_topk, k=far_rank)[0][-1]

            cond = torch.mul(torch.bitwise_not(topk_mask), torch.greater(tmp_cos_theta, cos_theta_neg_th))
            cos_theta_neg_topk = torch.mul(cond.to(torch.float32), tmp_cos_theta)
            cos_theta_neg_topk_ = torch.mul(cond.to(torch.float32), tmp_cos_theta_)
            cond = torch.greater(target_cos_theta_m, cos_theta_neg_topk)

            cos_theta_neg_topk = torch.where(cond, cos_theta_neg_topk, cos_theta_neg_topk_)
            cos_theta_neg_topk = torch.pow(cos_theta_neg_topk, 2)  # F = z^p = cos^2
            times = torch.sum(torch.greater(cos_theta_neg_topk, 0).to(torch.float32), dim=1, keepdim=True)
            times = torch.where(torch.greater(times, 0), times, torch.ones_like(times))
            cos_theta_neg_topk = torch.sum(cos_theta_neg_topk, dim=1, keepdim=True) / times  # ri+/ru+

            target_cos_theta_m = target_cos_theta_m - (1 + target_cos_theta_) * cos_theta_neg_topk
            cos_theta.scatter_(1, target_index.view(-1, 1).long(), target_cos_theta_m)
        output = cos_theta * self.scale
        loss = self.custom_loss(output, target_index, scale, explicit_kernel)
        acc = calc_top1_acc(origin_cos * self.scale, target_index,self.ddp)

        if return_logits:
            return loss, acc, output

        return loss, acc

    @staticmethod
    def min_max_normalize(x):
                min_val = torch.min(x)
                max_val = torch.max(x)
                return (x - min_val) / (max_val - min_val + 1e-8) 
    
    @staticmethod
    def kl_divergence(p, q):
            p = torch.clamp(p, min=1e-8)  
            q = torch.clamp(q, min=1e-8)  
            p = p / p.sum() 
            q = q / q.sum() 
            return torch.sum(p * torch.log(p / q))

    def compute_bank_loss(self, explicit_kernel):
        """The consistency loss between the implicit feature library and the explicit feature library
        Limit the IIE loss from distinguishing the implicit feature library beyond the explicit feature library
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        valid_mask = torch.ones(self.out_features, dtype=torch.bool, device=device)
        for idx in self.missing_ids:
            valid_mask[idx] = False
        random_vector = F.normalize(torch.randn(512, 1, device=device), dim=0)
        valid_implicit = self.kernel[:, valid_mask]  # [512, num_valid]
        valid_explicit = explicit_kernel[:, valid_mask]  # [512, num_valid]
        implicit_sims = F.cosine_similarity(random_vector, 
                                        F.normalize(valid_implicit, dim=0), dim=0)
        explicit_sims = F.cosine_similarity(random_vector, 
                                        F.normalize(valid_explicit, dim=0), dim=0)
        if torch.isnan(implicit_sims).any():
            print("Warning: NaN in implicit_sims")
            print(f"implicit_sims range: [{implicit_sims.min():.4f}, {implicit_sims.max():.4f}]")
        
        if torch.isnan(explicit_sims).any():
            print("Warning: NaN in explicit_sims")
            print(f"explicit_sims range: [{explicit_sims.min():.4f}, {explicit_sims.max():.4f}]")

        prob_implicit = self.min_max_normalize(implicit_sims)
        prob_explicit = self.min_max_normalize(explicit_sims)
        

        if torch.isnan(prob_implicit).any() or torch.isnan(prob_explicit).any():
            print("Warning: NaN in probability distributions")
            print("prob_implicit sum:", prob_implicit.sum().item())
            print("prob_explicit sum:", prob_explicit.sum().item())
        
        kl_loss = self.kl_divergence(prob_implicit, prob_explicit)
        if torch.isnan(kl_loss):
            print("Warning: KL loss is NaN!")
            return torch.tensor(0.0, device=device)
            
        return kl_loss


    def compute_FT_implicit_loss(self, valid_fake_embs, valid_target_indices, scale, valid_source_indices, em_kernel):
        """Calculate the similarity loss between the implicit feature of fake samples and the explicit feature of target samples
        Args:
            valid_fake_embs: [num_valid_fake, 512] - The implicit feature of the valid fake samples
            valid_target_indices: [num_valid_fake] - The corresponding target ID
            scale: float - The scaling factor
            valid_source_indices: [num_valid_fake] - The corresponding source ID
            em_kernel: [512, 1000] - The explicit feature library
        Returns:
            losses: [num_valid_fake] - The loss of each sample
        """
        num_samples = valid_fake_embs.size(0)
        losses = []

        for i in range(num_samples):
            fake_feat = F.normalize(valid_fake_embs[i], dim=0)  # [512]
            target_feat = F.normalize(em_kernel[:, valid_target_indices[i]], dim=0)  # [512]
            source_feat = F.normalize(em_kernel[:, valid_source_indices[i]], dim=0)  # [512]
            fake_target_sim = torch.dot(fake_feat, target_feat)
            source_target_sim = torch.dot(source_feat, target_feat)
            
            loss = torch.exp(scale * (1 - fake_target_sim))
            losses.append(loss)
        return torch.stack(losses)  # [num_valid_fake]


    def get_kernel_by_id(self, id):
        return self.new_kernel.get(id)
    
    def update_kernel(self, embeddings, target_indices, label, envID):
        """
        Update the sub-kernel in self.new_kernel based on the input embeddings, target_indices, and envID
        :param embeddings: The new feature, dimension is [batchsize, 512]
        :param target_indices: The targetID of each sample, dimension is [batchsize]
        label: [bs] - label (0: real, 1: fake)
        :param envID: The corresponding environment ID, dimension is [batchsize]
        """
        with torch.no_grad():
            valid_update_mask = (label == 0)
            for idx, (i, should_update) in enumerate(zip(target_indices, valid_update_mask)):
                if should_update:
                    target_id = target_indices[idx].item()
                    env_id = envID[idx].item()
                    embedding = embeddings[idx]
                    sub_kernel = self.get_kernel_by_id(target_id)
                    if sub_kernel is not None:
                        if env_id < sub_kernel.size(1):
                            sub_kernel[:, env_id] = embedding

            num_updated = valid_update_mask.sum().item()


    def compute_new_loss(self, valid_fake_embs, valid_target_indices, valid_source_indices, envID):
        losses = []
        for i in range(valid_fake_embs.size(0)):
            fake_feat = F.normalize(valid_fake_embs[i], dim=0).cuda()

            with torch.no_grad():
                target_feat = F.normalize(self.get_kernel_by_id((valid_target_indices[i]).item())[:, envID[i]], dim=0).detach().cuda()
                source_feat = F.normalize(self.get_kernel_by_id((valid_source_indices[i]).item())[:, 0], dim=0).detach().cuda()

            fake_source_sim = F.cosine_similarity(fake_feat, source_feat, dim=0)

            fake_target_sim = F.cosine_similarity(fake_feat, target_feat, dim=0)

            weighted_loss = 1 - fake_target_sim + torch.abs(fake_source_sim)

            losses.append(weighted_loss)

        return torch.stack(losses)
    

    def compute_IIIC_loss(self, im_embs, target_index, envID, em_kernel):
        losses = []
        for i in range(im_embs.size(0)):
            feat = im_embs[i]
            target_id = target_index[i]
            env_id = envID[i].item()
            with torch.no_grad():
                implicit_sub_kernel = self.get_kernel_by_id(target_id.item())
                if implicit_sub_kernel is None:
                    continue
                explicit_sub_kernel = em_kernel.get(target_id.item())
                if explicit_sub_kernel is None:
                    continue
            num_columns = implicit_sub_kernel.size(1)
            current_loss = 0
            for x in range(num_columns):
                if x == env_id:
                    continue
                implicit_similarity = F.cosine_similarity(feat.unsqueeze(0), F.normalize(implicit_sub_kernel[:, x].unsqueeze(0), p=2, dim=1).cuda())
                explicit_similarity = F.cosine_similarity(explicit_sub_kernel[:, env_id].unsqueeze(0).cuda(), explicit_sub_kernel[:, x].unsqueeze(0).cuda())
                if explicit_similarity.item() > 0.5:
                    current_loss += torch.abs(implicit_similarity - explicit_similarity)

            if isinstance(current_loss, torch.Tensor):
                losses.append(current_loss)
        
        return torch.stack(losses)


class FC_ddp(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 scale=8.0,
                 margin=0.2,
                 mode='cosface',
                 use_cifp=False,
                 reduction='mean'):
        """ Args:
            in_features: size of each input features
            out_features: size of each output features
            scale: norm of input feature
            margin: margin
        """
        super(FC_ddp, self).__init__()
        self.in_features = in_features
        self.out_features = out_features  # num of classes
        self.scale = scale
        self.margin = margin
        self.mode = mode
        self.use_cifp = use_cifp
        # self.kernel = Parameter(torch.Tensor(in_features, out_features))
        # nn.init.normal_(self.kernel, std=0.01)

        self.criteria = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.sig = torch.nn.Sigmoid()

    def apply_margin(self, target_cos_theta):
        assert self.mode in ['cosface', 'arcface'], 'Please check the mode'
        if self.mode == 'arcface':
            cos_m = math.cos(self.margin)
            sin_m = math.sin(self.margin)
            theta = math.cos(math.pi - self.margin)
            sinmm = math.sin(math.pi - self.margin) * self.margin
            sin_theta = torch.sqrt(1.0 - torch.pow(target_cos_theta, 2))
            cos_theta_m = target_cos_theta * cos_m - sin_theta * sin_m
            target_cos_theta_m = torch.where(
                target_cos_theta > theta, cos_theta_m, target_cos_theta - sinmm)
        elif self.mode == 'cosface':
            target_cos_theta_m = target_cos_theta - self.margin

        return target_cos_theta_m

    def forward(self, embeddings, label, return_logits=False):
        """

        :param embeddings: local gpu [bs, 512]
        :param label: local labels [bs]
        :param return_logits: bool
        :return:
        loss: computed local loss, w/wo CIFP
        acc: local accuracy in one gpu
        output: local logits with margins, with gradients, scaled, [bs, C].
        """
        sample_num = embeddings.size(0)
        cos_theta = self.sig(embeddings)
        target_cos_theta = cos_theta[torch.arange(0, sample_num), label].view(-1, 1)
        # target_cos_theta_m = target_cos_theta - self.margin
        target_cos_theta = target_cos_theta - self.margin
        # cos_theta.scatter_(1, label.view(-1, 1).long(), target_cos_theta_m)
        out = cos_theta.clone()
        out.scatter_(1, label.view(-1, 1).long(), target_cos_theta)

        out = out * self.scale

        loss = self.criteria(out, label)

        return loss