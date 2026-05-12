import torch
import torch.nn as nn
from easymocap.config import Config, load_object
from easymocap.mytools.debug_utils import log

def dict_of_numpy_to_tensor(body_params, device):
    params_ = {}
    for key, val in body_params.items():
        if isinstance(val, dict):
            params_[key] = dict_of_numpy_to_tensor(val, device)
        else:
            params_[key] = torch.Tensor(val).to(device)
    return params_

def dict_of_tensor_to_numpy(body_params):
    params_ = {}
    for key, val in body_params.items():
        if isinstance(val, dict):
            params_[key] = dict_of_tensor_to_numpy(val)
        else:
            params_[key] = val.cpu().numpy()
    return params_

def make_optimizer(opt_params, optim_type='lbfgs', max_iter=20,
    lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, **kwargs):
    if isinstance(opt_params, dict):
        # LBFGS 不支持参数字典
        opt_params = list(opt_params.values())
    if optim_type == 'lbfgs':
        # optimizer = torch.optim.LBFGS(
        #     opt_params, max_iter=max_iter, lr=lr, line_search_fn='strong_wolfe',
        #     tolerance_grad= 0.0000001, # float32的有效位数是7位
        #     tolerance_change=0.0000001,
        # )
        from easymocap.pyfitting.lbfgs import LBFGS
        optimizer = LBFGS(opt_params, line_search_fn='strong_wolfe', max_iter=max_iter,
                          tolerance_grad= 0.0000001, # float32的有效位数是7位
                            tolerance_change=0.0000001,
                          **kwargs)
    elif optim_type == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=0.01, betas=betas, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def grad_require(params, flag=False):
    if isinstance(params, list):
        for par in params:
            par.requires_grad = flag 
    elif isinstance(params, dict):
        for key, par in params.items():
            par.requires_grad = flag

def make_closure(optimizer, model, params, infos, loss, device):
    loss_func = {}
    for key, val in loss.items():
        loss_func[key] = load_object(val['module'], val['args'])
        if isinstance(loss_func[key], nn.Module):
            loss_func[key].to(device)
    
    # compiled_model = torch.compile(model)
    def closure(debug=False):
        torch.cuda.nvtx.range_push(f"closure_zero_grad")
        optimizer.zero_grad()
        new_params = params.copy()
        torch.cuda.nvtx.range_pop()


        torch.cuda.nvtx.range_push(f"model_forward_closure")
        output = model(new_params)
        torch.cuda.nvtx.range_pop()
        # output = compiled_model(new_params)
        loss_dict = {}
        loss_weight = {key:loss[key].weight for key in loss_func.keys()}
        torch.cuda.nvtx.range_push(f"loss_func_closure")
        for key, func in loss_func.items():
            # print(f"[closure] Computing loss: {key}")  # [closure] Computing loss: k3d
            # print(f"func: {func}")  # func: Keypoints3D()
            output_ = {k: output[k] for k in loss[key].key_from_output}
            infos_ = {k: infos[k] for k in loss[key].key_from_infos}
            loss_now = func(output_, infos_)
            if isinstance(loss_now, dict):
                for k, _loss in loss_now.items():
                    loss_dict[key+'_'+k] = _loss
                    loss_weight[key+'_'+k] = loss_weight[key]
                loss_weight.pop(key)
            else:
                loss_dict[key] = loss_now
        loss_sum = sum([loss_dict[key]*loss_weight[key]  
                        for key in loss_dict.keys()])     # loss_sum = 1000 * k3d_loss + 0.1 * regshape_loss + ...
        torch.cuda.nvtx.range_pop()
        # for key in loss_dict.keys():
        #     print(key, loss_dict[key] * loss_weight[key])
        # print(loss_sum)
        if debug:
            return loss_dict, loss_weight
        torch.cuda.nvtx.range_push("closure_backward")
        loss_sum.backward()
        torch.cuda.nvtx.range_pop()
        return loss_sum
    return closure

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([1e-5, abs(prev_val), abs(curr_val)])

class Optimizer:
    def __init__(self, optimize_keys, optimizer_args, loss) -> None:
        self.optimize_keys = optimize_keys
        self.optimizer_args = optimizer_args
        self.loss = loss
        self.used_infos = []
        for key, val in loss.items():
            self.used_infos.extend(val.key_from_infos)
        self.used_infos = list(set(self.used_infos))
        self.iter = 0
        torch.cuda.empty_cache()

    def log_loss(self, iter_, closure, print_loss=False):
        if iter_ % 10 == 0 or print_loss:
            with torch.no_grad():
                loss_dict, loss_weight = closure(debug=True)
            print('{:-6d}: '.format(iter_) + ' '.join([key + ' %7.4f'%(loss_dict[key].item()*loss_weight[key]) for key in loss_dict.keys()]))
        
    def optimizer_step(self, optimizer, closure):
        prev_loss = None
        self.log_loss(0, closure, True)
        for iter_ in range(1, 1000):
            torch.cuda.nvtx.range_push(f"LBFGS_iter{iter_}")
            loss = optimizer.step(closure)
            torch.cuda.nvtx.range_pop()
            # check the loss
            if torch.isnan(loss).sum() > 0:
                print('[optimize] NaN loss value, stopping!')
                break
            if torch.isinf(loss).sum() > 0:
                print('[optimize] Infinite loss value, stopping!')
                break
            # check the delta
            if iter_ > 0 and prev_loss is not None:
                loss_rel_change = rel_change(prev_loss, loss.item())
                if loss_rel_change <= 0.0000001:
                    break
            self.log_loss(iter_, closure)
            prev_loss = loss.item()
        print(f'[optimize] Finished at iter {iter_}\n')
        self.log_loss(iter_, closure, True)
        return True


    def __call__(self, params, model, **infos):
        """
            待优化变量一定要在params中,但params中不一定会被优化
            infos中的变量不一定会被优化
        """
        # TODO: 应该使用model的device，但考虑到model可能是一个函数，所以暂时当场计算
        torch.cuda.nvtx.range_push(f"prepare_optimizer_inputs")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        params = dict_of_numpy_to_tensor(params, device=device)  #把初始好的smpl參數包成tensor:  dict_keys(['Rh', 'Th', 'poses', 'shapes'])
        infos_used = {key: infos[key] for key in self.used_infos if key in infos.keys()} 
        infos_used = dict_of_numpy_to_tensor(infos_used, device=device)   # dict_keys(['keypoints3d'])
        
        
        optimize_keys = self.optimize_keys  # [['poses', 'Rh', 'Th'], ['poses', 'shapes', 'Rh', 'Th']]

        if isinstance(optimize_keys[0], list):
            optimize_keys = optimize_keys[self.iter]  # ['poses', 'Rh', 'Th']

        log('[{}] Optimize {}'.format(self.__class__.__name__, optimize_keys))
        log('[{}] Loading {}'.format(self.__class__.__name__, self.used_infos)) # # dict_keys(['keypoints3d'])
        opt_params = {}
        for key in optimize_keys:
            if key in infos.keys(): # 优化的参数
                opt_params[key] = infos_used[key]
            elif key in params.keys():
                opt_params[key] = params[key]
            else:
                raise ValueError('{} is not in infos or body_params'.format(key))
        for key, val in opt_params.items():
            infos_used['init_'+key] = val.clone()
        torch.cuda.nvtx.range_pop()
        # print(infos_used) 
        # {'keypoints3d': tensor([[ 0.09622, -1.52626,  0.60554,  0.91292]....], device='cuda:0'), 'init_shapes': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')}
        # print(opt_params) # {'shapes': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')}
        
        # args:
        #     optimizer_args: {optim_type: lbfgs}
        #     optimize_keys: [shapes]
        #     loss:
        #     k3d:
        #         weight: 1000.
        #         module: myeasymocap.operations.loss.LimbLength
        #         key_from_output: [keypoints]
        #         key_from_infos: [keypoints3d]
        #         args:
        #         kintree: [[8, 1], [2, 5], [2, 3], [5, 6], [3, 4], [6, 7], [2, 3], [5, 6], [3, 4], [6, 7], [2, 3], [5, 6], [3, 4], [6, 7], [1, 0], [9, 12], [9, 10], [10, 11], [12, 13],[13, 14]]
        #     regshape:
        #         weight: 0.1
        #         module: myeasymocap.operations.loss.RegLoss
        #         key_from_output: [shapes]
        #         key_from_infos: [] # TODO: 根据2D的置信度来计算smooth权重
        #         args:
        #         key: shapes
        #         norm: l2
        torch.cuda.nvtx.range_push(f"make_optimizer")
        optimizer = make_optimizer(opt_params, **self.optimizer_args)  # 建立 optimizer
        torch.cuda.nvtx.range_pop()
        
        
        # compiled_model = torch.compile(model)
        # closure = make_closure(optimizer, compiled_model, params, infos_used, self.loss, device)
        # print(model)
        # print(params)
        torch.cuda.nvtx.range_push(f"make_closure")
        closure = make_closure(optimizer, model, params, infos_used, self.loss, device)  # 建立 closure 函數（計算 loss）
        torch.cuda.nvtx.range_pop()
        # 准备开始优化
        torch.cuda.nvtx.range_push(f"grad_require")
        grad_require(opt_params, True) # 打開參數的梯度開關 (requires_grad=True)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"optimizer_step")
        self.optimizer_step(optimizer, closure)  # 執行 optimizer_step() 優化迴圈
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"grad_require")
        grad_require(opt_params, False) # 關閉梯度 (requires_grad=False)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"prepare_return_values")
        # 直接返回
        ret = {
            'params': params
        }
        for key in optimize_keys:
            if key in infos.keys():
                ret[key] = opt_params[key]
        ret = dict_of_tensor_to_numpy(ret)
        torch.cuda.nvtx.range_pop()
        return ret