import torch
import torch.nn as nn
from .base import Model
from .smpl import SMPLModel, SMPLLayerEmbedding, read_pickle, to_tensor, to_np
from os.path import join
import numpy as np

def read_hand(path, use_pca, use_flat_mean, num_pca_comps):
    data = read_pickle(path)
    mean = data['hands_mean'].reshape(1, -1).astype(np.float32)
    mean_full = mean
    components_full = data['hands_components'].astype(np.float32)
    weight = np.diag(components_full @ components_full.T)
    components = components_full[:num_pca_comps]
    weight = weight[:num_pca_comps]
    if use_flat_mean:
        mean = np.zeros_like(mean)
    return mean, components, weight, mean_full, components_full

class SMPLXModel(SMPLModel):
    def __init__(self, model_path, num_pca_comps=12, use_pca=True,
    regressor_path=None, num_expression_coeffs=10, **kwargs):

        # ── Must be assigned BEFORE super().__init__() ────────────────────────
        # super().__init__() calls self.register_any_lbs() (overridden below).
        # That method reads self.num_betas and self.num_expression_coeffs,
        # so they must exist at that point.
        num_betas = kwargs.pop('NUM_SHAPES', 10)   # body-shape PCs (e.g. 10)
        self.num_betas             = num_betas
        self.num_expression_coeffs = num_expression_coeffs

        # Tell parent the combined width; our register_any_lbs fills it correctly.
        kwargs['NUM_SHAPES'] = num_betas + num_expression_coeffs  # e.g. 20

        super().__init__(model_path=model_path, regressor_path=regressor_path, **kwargs)
        # After super().__init__():
        #   self.shapedirs   (V, 3, 20) ← body [0:10] + expr [300:310]  (correct)
        #   self.j_shapedirs (K, 3, 20) ← built from the corrected shapedirs

        self.model_type    = 'smplx'
        self.use_pca       = use_pca
        self.num_pca_comps = num_pca_comps

        # ── Hand PCA components (buffers → follow .to(device)) ────────────────
        if 'hands_componentsl' in self.data:
            comp_l = self.data['hands_componentsl'][:num_pca_comps]
            mean_l = self.data['hands_meanl']
        else:
            comp_l = np.eye(45)[:num_pca_comps]
            mean_l = np.zeros(45)
        self.register_buffer('hand_components_l', to_tensor(comp_l, dtype=torch.float32))
        self.register_buffer('hand_mean_l',       to_tensor(mean_l, dtype=torch.float32))

        if 'hands_componentsr' in self.data:
            comp_r = self.data['hands_componentsr'][:num_pca_comps]
            mean_r = self.data['hands_meanr']
        else:
            comp_r = np.eye(45)[:num_pca_comps]
            mean_r = np.zeros(45)
        self.register_buffer('hand_components_r', to_tensor(comp_r, dtype=torch.float32))
        self.register_buffer('hand_mean_r',       to_tensor(mean_r, dtype=torch.float32))

    # ── Core fix: load shapedirs from the correct regions of the 400-dim axis ─

    def register_any_lbs(self, data):
        """Override to combine body-shape and expression shapedirs correctly.

        SMPLX_NEUTRAL.npz layout (400 dims along axis -1):
            [  0 : 300 ]  body-shape PCA components
            [300 : 400 ]  expression PCA components

        The parent naively does shapedirs[..., :NUM_SHAPES] = [..., :20],
        which picks body PCs 1-20 — expression PCs never appear.

        We instead extract:
            body_sd = shapedirs[..., :num_betas]              (V, 3, 10)
            expr_sd = shapedirs[..., 300:300+num_expression]  (V, 3, 10)
        and concatenate them so that in forward():
            shapes     (B, 10)  activates dims  0-9  → body PCs
            expression (B, 10)  activates dims 10-19 → expression PCs
        """
        # Let parent register faces, J_regressor, v_template, weights,
        # posedirs, parents — and also write a temporarily wrong shapedirs.
        super().register_any_lbs(data)

        raw_sd  = to_np(data['shapedirs'])                              # (V, 3, 400)
        body_sd = raw_sd[..., :self.num_betas]                          # (V, 3, 10)
        expr_sd = raw_sd[..., 300 : 300 + self.num_expression_coeffs]   # (V, 3, 10)
        combined = np.concatenate([body_sd, expr_sd], axis=-1)           # (V, 3, 20)

        # Overwrite with correctly assembled tensor.
        self.register_buffer('shapedirs', to_tensor(combined, dtype=self.dtype))
        # register_any_keypoints() runs next inside super().__init__() and
        # will build j_shapedirs from this corrected self.shapedirs automatically.

    def stitch_pose(self, params):
        poses = params['poses']
        
        root_pose = poses[:, 0:3]
        body_pose = poses[:, 3:66]
        jaw_pose  = poses[:, 66:69]
        eye_pose  = poses[:, 69:75]
        
        # Branch based on whether we are using compressed hands or full 133-point COCO articulation
        if self.use_pca:
            handl = poses[:, 75:75+self.num_pca_comps]
            handr = poses[:, 75+self.num_pca_comps:75+self.num_pca_comps*2]
            
            handl_full = torch.matmul(handl, self.hand_components_l) + self.hand_mean_l
            handr_full = torch.matmul(handr, self.hand_components_r) + self.hand_mean_r
        else:
            # When use_pca=False (num_poses=165): full 45-DoF articulation.
            # forward() already cast all params to tensors, so poses is always
            # a tensor here. Just add the learned mean offset.
            handl_full = poses[:, 75:120]  + self.hand_mean_l
            handr_full = poses[:, 120:165] + self.hand_mean_r

        pose_parts = [root_pose, body_pose, jaw_pose, eye_pose, handl_full, handr_full]
        tensor_parts = []
        
        for part in pose_parts:
            if not torch.is_tensor(part):
                # Convert NumPy arrays or lists to Tensors matching the model's device
                part = torch.tensor(np.array(part), dtype=self.hand_mean_l.dtype, device=self.hand_mean_l.device)
            # JSON arrays sometimes lose their batch dimension; ensure they are 2D
            if part.dim() == 1:
                part = part.unsqueeze(0)
            tensor_parts.append(part)
            
        full_pose = torch.cat(tensor_parts, dim=1)
        return full_pose

    def forward(self, params=None, return_verts=True, return_tensor=True, **kwargs):
        # Prevent old NumPy arrays in kwargs from overwriting our new tensors later
        if params is None:
            params = kwargs.copy()
            kwargs = {} 
        else:
            params = params.copy()

        # --- THE UNIVERSAL TYPE-CASTING BLOCK ---
        # Intercept ALL incoming parameters and force to 2-D tensors on model device.
        # Iterate over a snapshot (list) so we can safely reassign values in the dict.
        for key, val in list(params.items()):
            if val is not None and not torch.is_tensor(val):
                val_t = torch.tensor(np.array(val), dtype=self.hand_mean_l.dtype, device=self.hand_mean_l.device)
                if val_t.dim() == 1:
                    val_t = val_t.unsqueeze(0)
                params[key] = val_t
            elif torch.is_tensor(val) and val.dim() == 1:
                params[key] = val.unsqueeze(0)
        # ----------------------------------------
        # ── Merge expression into shapes before LBS ──────────────────────────────
        # shapes  : (1 or B, num_betas)           — body shape only
        # expression: (B, num_expression_coeffs)  — per-frame expression
        # combined: (B, num_betas + num_expression_coeffs) fed to blend_shapes()
        if 'expression' in params and params['expression'] is not None:
            shapes     = params['shapes']
            expression = params['expression']
            # Expand shapes from (1, num_betas) to (B, num_betas) when shared
            bn = expression.shape[0]
            if shapes.shape[0] < bn:
                shapes = shapes.expand(bn, -1)
            # Guard against double-concatenation (e.g. second forward call)
            if shapes.shape[1] == self.num_betas:
                params['shapes'] = torch.cat([shapes, expression], dim=1)
            # If shapes is already (B, num_betas + num_expression_coeffs), leave it
        # ─────────────────────────────────────────────────────────────────────────
        full_pose = self.stitch_pose(params)
        params_full = params.copy()
        params_full['poses'] = full_pose
        
        # Note: **kwargs is empty here if we unpacked it, saving us from the overwrite bug!
        return super().forward(return_verts=return_verts, return_tensor=return_tensor, **params_full, **kwargs)

    def keypoints(self, params, return_tensor=True):
        return self.forward(params, return_verts=False, return_tensor=return_tensor)

class MANO(SMPLModel):
    def __init__(self, cfg_hand, **kwargs):
        super().__init__(**kwargs)
        self.name = 'mano'
        self.use_root_rot = False
        mean, components, weight, mean_full, components_full = read_hand(kwargs['model_path'], **cfg_hand)
        self.register_buffer('mean', to_tensor(mean, dtype=self.dtype))
        self.register_buffer('components', to_tensor(components, dtype=self.dtype))
        self.cfg_hand = cfg_hand
        self.to(self.device)
        if cfg_hand.use_pca:
            self.NUM_POSES = cfg_hand.num_pca_comps
    
    def extend_poses(self, poses, **kwargs):
        if poses.shape[-1] == self.mean.shape[-1] + 3:
            return poses
        if self.cfg_hand.use_pca:
            poses = poses @ self.components
        if kwargs.get('pose2rot', True):
            poses = super().extend_poses(poses+self.mean, **kwargs)
        else:
            poses = super().extend_poses(poses, **kwargs)
        return poses
    
    def jacobian_posesfull_poses(self, poses, poses_full):
        if self.cfg_hand.use_pca:
            jacobian = self.components.t()
            zero_root = torch.zeros((3, poses.shape[-1]), dtype=poses.dtype, device=poses.device)
            jacobian = torch.cat([zero_root, jacobian], dim=0)
        else:
            jacobian = super().jacobian_posesfull_poses(poses, poses_full)
        return jacobian
class MANOLR(Model):
    def __init__(self, model_path, regressor_path, cfg_hand, **kwargs):
        super().__init__()
        self.name = 'manolr'
        keys = list(model_path.keys())
        # stack 方式：(nframes, nhand x ndim)
        self.keys = keys
        modules_hand = {}
        faces = []
        v_template = []
        cnt = 0
        for key in keys:
            modules_hand[key] = MANO(cfg_hand, model_path=model_path[key], regressor_path=regressor_path[key], **kwargs)
            v_template.append(modules_hand[key].v_template.cpu().numpy())
            faces.append(modules_hand[key].faces + cnt)
            cnt += v_template[-1].shape[0]
            self.device = modules_hand[key].device
            self.dtype = modules_hand[key].dtype
            if key == 'right':
                modules_hand[key].shapedirs[:, 0] *= -1
                modules_hand[key].j_shapedirs[:, 0] *= -1
        self.faces = np.vstack(faces)
        self.v_template = np.vstack(v_template)
        self.modules_hand = nn.ModuleDict(modules_hand)
        self.to(self.device)

    def init_params(self, **kwargs):
        param_all = {}
        for key in self.keys:
            param = self.modules_hand[key].init_params(**kwargs)
            param_all[key] = param
        if False:
            params = {k: torch.cat([param_all[key][k] for key in self.keys], dim=-1) for k in param.keys()}
        else:
            params = {k: np.concatenate([param_all[key][k] for key in self.keys], axis=-1) for k in param.keys() if k != 'shapes'}
            params['shapes'] = param_all['left']['shapes']
        return params

    def split(self, params):
        params_split = {}
        for imodel, model in enumerate(self.keys):
            param_= params.copy()
            for key in ['poses', 'shapes', 'Rh', 'Th']:
                if key not in params.keys():continue
                if key == 'shapes':
                    continue
                shape = params[key].shape[-1]
                start = shape//len(self.keys)*imodel
                end = shape//len(self.keys)*(imodel+1)
                param_[key] = params[key][:, start:end]
            params_split[model] = param_
        return params_split

    def forward(self, **params):
        params_split = self.split(params)
        rets = []
        for imodel, model in enumerate(self.keys):
            ret = self.modules_hand[model](**params_split[model])
            rets.append(ret)
        if params.get('return_tensor', True):
            rets = torch.cat(rets, dim=1)
        else:
            rets = np.concatenate(rets, axis=1)
        return rets

    def extend_poses(self, poses, **kwargs):
        params_split = self.split({'poses': poses})
        rets = []
        for imodel, model in enumerate(self.keys):
            poses = params_split[model]['poses']
            poses = self.modules_hand[model].extend_poses(poses)
            rets.append(poses)
        poses = torch.cat(rets, dim=1)
        return poses

    def export_full_poses(self, poses, **kwargs):
        params_split = self.split({'poses': poses})
        rets = []
        for imodel, model in enumerate(self.keys):
            poses = torch.Tensor(params_split[model]['poses']).to(self.device)
            poses = self.modules_hand[model].extend_poses(poses)
            rets.append(poses)
        poses = torch.cat(rets, dim=1)
        return poses.detach().cpu().numpy()

class SMPLHModel(SMPLModel):
    def __init__(self, mano_path, cfg_hand, **kwargs):
        super().__init__(**kwargs)
        self.NUM_POSES = self.NUM_POSES - 90
        meanl, componentsl, weight_l, self.mean_full_l, self.components_full_l = read_hand(join(mano_path, 'MANO_LEFT.pkl'), **cfg_hand)
        meanr, componentsr, weight_r, self.mean_full_r, self.components_full_r = read_hand(join(mano_path, 'MANO_RIGHT.pkl'), **cfg_hand)
        self.register_buffer('weight_l', to_tensor(weight_l, dtype=self.dtype))
        self.register_buffer('weight_r', to_tensor(weight_r, dtype=self.dtype))
        self.register_buffer('meanl', to_tensor(meanl, dtype=self.dtype))
        self.register_buffer('meanr', to_tensor(meanr, dtype=self.dtype))
        self.register_buffer('componentsl', to_tensor(componentsl, dtype=self.dtype))
        self.register_buffer('componentsr', to_tensor(componentsr, dtype=self.dtype))

        self.register_buffer('jacobian_posesfull_poses_', self._jacobian_posesfull_poses())
        self.NUM_HANDS = cfg_hand.num_pca_comps if cfg_hand.use_pca else 45
        self.cfg_hand = cfg_hand
        self.to(self.device)
    
    def _jacobian_posesfull_poses(self):
        # TODO: cache this 
        # | body_full/body | 0 | 0 |
        # |      0         | l | 0 |
        # |      0         | 0 | r |
        eye_right = torch.eye(self.NUM_POSES, dtype=self.dtype)
        # 
        jac_handl = self.componentsl.t()
        jac_handr = self.componentsr.t()
        output = torch.zeros((self.NUM_POSES_FULL, self.NUM_POSES+jac_handl.shape[1]*2), dtype=self.dtype)
        if self.use_root_rot:
            raise NotImplementedError
        else:
            output[3:3+self.NUM_POSES, :self.NUM_POSES] = eye_right
            output[3+self.NUM_POSES:3+self.NUM_POSES+jac_handl.shape[0], \
                self.NUM_POSES:self.NUM_POSES+jac_handl.shape[1]] = jac_handl
            output[3+self.NUM_POSES+jac_handl.shape[0]:3+self.NUM_POSES+2*jac_handl.shape[0], \
                self.NUM_POSES+jac_handl.shape[1]:self.NUM_POSES+jac_handl.shape[1]*2] = jac_handr            
        return output

    def init_params(self, nFrames=1, nShapes=1, nPerson=1, ret_tensor=False, add_scale=False):
        params = super().init_params(nFrames, nShapes, nPerson, ret_tensor, add_scale=add_scale)
        handl = np.zeros((nFrames, self.NUM_HANDS))
        handr = np.zeros((nFrames, self.NUM_HANDS))
        if nPerson > 1:
            handl = handl[:, None].repeat(nPerson, axis=1)
            handr = handr[:, None].repeat(nPerson, axis=1)
        if ret_tensor:
            handl = to_tensor(handl, self.dtype, self.device)
            handr = to_tensor(handr, self.dtype, self.device)
        params['handl'] = handl
        params['handr'] = handr
        return params
    
    def extend_poses(self, poses, handl=None, handr=None, **kwargs):
        if poses.shape[-1] == self.NUM_POSES_FULL:
            return poses
        poses = super().extend_poses(poses)
        if handl is None:
            handl = self.meanl.clone()
            handr = self.meanr.clone()
            handl = handl.expand(poses.shape[0], -1)
            handr = handr.expand(poses.shape[0], -1)
        else:
            if self.cfg_hand.use_pca:
                handl = handl @ self.componentsl
                handr = handr @ self.componentsr
            handl = handl +self.meanl
            handr = handr +self.meanr
        poses = torch.cat([poses, handl, handr], dim=-1)
        return poses
    
    def export_full_poses(self, poses, handl, handr, **kwargs):
        poses = torch.Tensor(poses).to(self.device)
        handl = torch.Tensor(handl).to(self.device)
        handr = torch.Tensor(handr).to(self.device)
        poses = self.extend_poses(poses, handl, handr)
        return poses.detach().cpu().numpy()

class SMPLHModelEmbedding(SMPLHModel):
    def __init__(self, vposer_ckpt='data/body_models/vposer_v02', **kwargs):
        super().__init__(**kwargs)
        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser
        vposer, _ = load_model(vposer_ckpt, 
            model_code=VPoser,
            remove_words_in_model_weights='vp_model.',
            disable_grad=True)
        vposer.to(self.device)
        self.vposer = vposer
        self.vposer_dim = 32
        self.NUM_POSES = self.vposer_dim
    
    def decode(self, poses, add_rot=True):
        if poses.shape[-1] == 66 and add_rot:
            return poses
        elif poses.shape[-1] == 63 and not add_rot:
            return poses
        assert poses.shape[-1] == self.vposer_dim, poses.shape
        ret = self.vposer.decode(poses)
        poses_body = ret['pose_body'].reshape(poses.shape[0], -1)
        if add_rot:
            zero_rot = torch.zeros((poses.shape[0], 3), dtype=poses.dtype, device=poses.device)
            poses_body = torch.cat([zero_rot, poses_body], dim=-1)
        return poses_body

    def extend_poses(self, poses, handl, handr, **kwargs):
        if poses.shape[-1] == self.NUM_POSES_FULL:
            return poses
        zero_rot = torch.zeros((poses.shape[0], 3), dtype=poses.dtype, device=poses.device)
        poses_body = self.decode(poses, add_rot=False)
        if self.cfg_hand.use_pca:
            handl = handl @ self.componentsl
            handr = handr @ self.componentsr
        handl = handl +self.meanl
        handr = handr +self.meanr
        poses = torch.cat([zero_rot, poses_body, handl, handr], dim=-1)
        return poses
    
    def export_full_poses(self, poses, handl, handr, **kwargs):
        poses = torch.Tensor(poses).to(self.device)
        handl = torch.Tensor(handl).to(self.device)
        handr = torch.Tensor(handr).to(self.device)
        poses = self.extend_poses(poses, handl, handr)
        return poses.detach().cpu().numpy()
