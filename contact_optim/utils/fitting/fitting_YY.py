import torch
import numpy as np

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])

@torch.no_grad()
class FittingMonitor():
    def __init__(self, summary_steps=1,
                 maxiters=30, ftol=1e-09, gtol=1e-09,
                 model_type='smpl'):
        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.summary_steps = summary_steps
        self.model_type = model_type

    def run_fitting(self, optimizer, closure, params, body_models,
                    use_vposer=True, pose_embeddings=None, vposer=None, use_motionprior=False, cameras=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        #append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        print('\n')
        for n in range(self.maxiters):
            loss, loss_dict = optimizer.step(closure)
            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            prev_loss = loss.item()
            # print('stage fitting loss: ', prev_loss)
            print('stage fitting loss: %.1f' % prev_loss, loss_dict)
        return prev_loss

    def create_fitting_closure(self, optimizer, body_models, transl, pose, shape, joint2ds, regressor, intri,
                               create_graph=False):

        def fitting_func(backward=True):
            loss_dict = {}
            total_loss = 0.
            if backward:
                optimizer.zero_grad()

            pred_meshes, _ = body_models(shape, pose, transl)
            pred_joints = torch.matmul(regressor, pred_meshes)
            n_frame, n_joint, _ = pred_joints.shape

            proj2ds = torch.matmul(intri, pred_joints.reshape(-1, 3).T).T
            proj2ds = proj2ds[:,:2] / proj2ds[:,2:]
            proj2ds = proj2ds.reshape(n_frame, n_joint, -1)


            reproj_loss = (torch.norm(proj2ds-joint2ds[:,:,:2], dim=-1) * joint2ds[:,:,2]).mean()
            smooth_loss = torch.norm(transl[1:] - transl[:-1], dim=-1).mean() * 100

            loss_dict['reproj_loss'] = reproj_loss
            loss_dict['smooth_loss'] = smooth_loss

            for key in loss_dict:
                total_loss += loss_dict[key]
                loss_dict[key] = float(loss_dict[key].data)

            if backward:
                total_loss.backward(retain_graph=False, create_graph=create_graph)

            return total_loss, loss_dict

        return fitting_func


