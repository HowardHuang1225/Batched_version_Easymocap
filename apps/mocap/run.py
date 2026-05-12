# 这个脚本提供mocap的基本运行接口
import os
from easymocap.config import Config, load_object
from tqdm import tqdm
import time
import torch
import numpy as np

def process(dataset, model, args):
    ret_all = []
    print('[Run] dataset has {} samples'.format(len(dataset)))

    # 記錄各子模組累計時間
    time_records_step = {k: 0.0 for k in model.model_steps.keys()}
    time_records_final = {}  # final 各子模組累計耗時
    total_final_time = 0.0

    # ===== 新增全局 bbox 統計 =====
    all_bboxes_list = []

    for i in tqdm(range(len(dataset)), desc='[Run]'):
        print(f"\n========== START DATA LOADING (SAMPLE {i}) ==========")
        start = time.perf_counter()
        data = dataset[i]
        time_records_step['data_loading'] = time_records_step.get('data_loading', 0.0) + (time.perf_counter() - start)

        print(f"\n========== START MODEL STEP (SAMPLE {i}) ==========")
        start_step = time.perf_counter()
        ret = model.at_step(data, i)

        # ===== 收集 detect 的 bbox =====
        # if 'bbox' in ret:
        #     bboxes_list = ret['bbox']
        #     for arr in bboxes_list:
        #         if arr.shape[0] > 0:
        #             all_bboxes_list.append(arr[:, :4])  # 只取 x1,y1,x2,y2

        # 細分每個 step 子模組耗時
        for k, t in model.last_step.items():
            time_records_step[k] = time_records_step.get(k, 0.0) + t
        time_records_step['at_step_total'] = time_records_step.get('at_step_total', 0.0) + (time.perf_counter() - start_step)

        if not args.skip_final:
            ret_all.append(ret)

    # ===== 計算全局 bbox 面積平均 =====
        # ===== 計算全局 bbox 面積統計 =====
    # if len(all_bboxes_list) == 0:
    #     global_bbox_count = 0
    #     global_bbox_area_mean = 0
    #     global_bbox_area_min = 0
    #     global_bbox_area_max = 0
    # else:
    #     all_bboxes = np.concatenate(all_bboxes_list, axis=0)
    #     all_bboxes = torch.from_numpy(all_bboxes).float()
    #     areas = (all_bboxes[:, 2] - all_bboxes[:, 0]) * (all_bboxes[:, 3] - all_bboxes[:, 1])
    #     global_bbox_count = all_bboxes.shape[0]
    #     global_bbox_area_mean = areas.mean().item()
    #     global_bbox_area_min = areas.min().item()
    #     global_bbox_area_max = areas.max().item()

    # print(f"\n[Detect Stats] Total bbox: {global_bbox_count}, "
    #     f"Mean area: {global_bbox_area_mean:.2f}, "
    #     f"Min area: {global_bbox_area_min:.2f}, "
    #     f"Max area: {global_bbox_area_max:.2f}")


    if not args.skip_final:
        print("\n========== START FINAL PROCESS ==========")
        start_final = time.perf_counter()
        ret_all = model.at_final(ret_all)
        total_final_time += time.perf_counter() - start_final

        # 細分 final 子模組耗時
        for k, t in model.last_final.items():
            time_records_final[k] = time_records_final.get(k, 0.0) + t

    # 印出 step 總耗時
    print("\n[Timing] Total time per step stage:")
    for k, t in time_records_step.items():
        print(f"{k:20s}: {t:.3f}s")

    # 印出 final 各子模組耗時
    if not args.skip_final:
        print("\n[Timing] Total time per final stage:")
        for k, t in time_records_final.items():
            print(f"{k:20s}: {t:.3f}s")
        print(f"{'at_final_total':20s}: {total_final_time:.3f}s")


def process_batched(dataset, model, args, batch_size=4):
    ret_all = []
    print('[Run] dataset has {} samples'.format(len(dataset)))

    # 記錄各子模組累計時間
    time_records_step = {k: 0.0 for k in model.model_steps.keys()}
    time_records_final = {}
    total_final_time = 0.0

    for start_idx in tqdm(range(0, len(dataset), batch_size), desc='[Run]'):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_indices = list(range(start_idx, end_idx))

        print(f"\n========== START DATA LOADING (SAMPLES {start_idx}:{end_idx}) ==========")
        start_step = time.perf_counter()  # 從 data loading 開始計時

        torch.cuda.nvtx.range_push(f"data_loading_{start_idx}_{end_idx}_batch")
        batch_data = [dataset[i] for i in batch_indices]
        torch.cuda.nvtx.range_pop()

        # 記錄 data loading 時間
        time_records_step['data_loading'] = (
            time_records_step.get('data_loading', 0.0)
            + (time.perf_counter() - start_step)
        )

        print(f"\n========== START MODEL STEP (SAMPLES {start_idx}:{end_idx}) ==========")
        
        torch.cuda.nvtx.range_push(f"model_at_step_{start_idx}_{end_idx}_batch")
        ret_batch = model.at_step(batch_data, batch_indices)
        torch.cuda.nvtx.range_pop()

        if not isinstance(ret_batch, list):
            ret_batch = [ret_batch]

        if isinstance(model.last_step, list):
            for step_timer in model.last_step:
                for k, t in step_timer.items():
                    time_records_step[k] = time_records_step.get(k, 0.0) + t
        else:
            for k, t in model.last_step.items():
                time_records_step[k] = time_records_step.get(k, 0.0) + t

        # 把整個 batch 的總時間（包含 data loading + model step）累加到 at_step_total
        time_records_step['at_step_total'] = (
            time_records_step.get('at_step_total', 0.0)
            + (time.perf_counter() - start_step)
        )

        if not args.skip_final:
            ret_all.extend(ret_batch)


    # ===== Final 阶段 =====
    if not args.skip_final:
        print("\n========== START FINAL PROCESS ==========")
        start_final = time.perf_counter()
        torch.cuda.nvtx.range_push("model_at_final")
        ret_all = model.at_final(ret_all)
        torch.cuda.nvtx.range_pop()
        total_final_time += time.perf_counter() - start_final

        for k, t in model.last_final.items():
            time_records_final[k] = time_records_final.get(k, 0.0) + t

    # ===== 印出時間統計 =====
    print("\n[Timing] Total time per step stage:")
    for k, t in time_records_step.items():
        print(f"{k:20s}: {t:.3f}s")

    if not args.skip_final:
        print("\n[Timing] Total time per final stage:")
        for k, t in time_records_final.items():
            print(f"{k:20s}: {t:.3f}s")
        print(f"{'at_final_total':20s}: {total_final_time:.3f}s")




def update_data_by_args(cfg_data, args):
    if args.root is not None:
        cfg_data.args.root = args.root
    if args.subs is not None:
        cfg_data.args.subs = args.subs
    if args.subs_vis is not None:
        cfg_data.args.subs_vis = args.subs_vis
    if args.ranges is not None:
        cfg_data.args.ranges = args.ranges
    if args.cameras is not None:
        cfg_data.args.reader.cameras.root = args.cameras
    if args.skip_vis or args.skip_vis_step:
        cfg_data.args.subs_vis = []
    return cfg_data

def update_exp_by_args(cfg_exp, args):
    opts_alias = []
    if 'alias' in cfg_exp.keys():
        for i in range(len(args.opt_exp)//2):
            if args.opt_exp[i*2] in cfg_exp.alias.keys():
                opts_alias.append(cfg_exp.alias[args.opt_exp[i*2]])
                opts_alias.append(args.opt_exp[i*2+1])
        cfg_exp.merge_from_list(opts_alias)
    if args.skip_vis or args.skip_vis_step:
        for key, val in cfg_exp.args.at_step.items():
            if key.startswith('vis'):
                val.skip = True
    if args.skip_vis or args.skip_vis_final:
        for key, val in cfg_exp.args.at_final.items():
            if key.startswith('vis') or key == 'make_video':
                val.skip = True    

def load_cfg_from_file(cfg, args):
    cfg = Config.load(cfg)
    cfg_data = Config.load(cfg.data)
    cfg_data.args.merge_from_other_cfg(cfg.data_opts)
    cfg_data = update_data_by_args(cfg_data, args)
    cfg_exp = Config.load(cfg.exp)
    cfg_exp.args.merge_from_other_cfg(cfg.exp_opts)
    update_exp_by_args(cfg_exp, args)
    return cfg_data, cfg_exp

def load_cfg_from_cmd(args):
    cfg_data = Config.load(args.data, args.opt_data)
    cfg_data = update_data_by_args(cfg_data, args)
    cfg_exp = Config.load(args.exp, args.opt_exp)
    update_exp_by_args(cfg_exp, args)
    return cfg_data, cfg_exp

def main_entrypoint():
    start_time = time.time()
    start_time1 = time.time()
    torch.cuda.nvtx.range_push(f"Configuration")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    for name in ['data', 'exp']:
        parser.add_argument('--{}'.format(name), type=str, required=False)
        parser.add_argument('--opt_{}'.format(name), type=str, nargs='+', default=[])
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--subs', type=str, default=None, nargs='+')
    parser.add_argument('--subs_vis', type=str, default=None, nargs='+')
    parser.add_argument('--ranges', type=int, default=None, nargs=3)
    parser.add_argument('--cameras', type=str, default=None, help='Camera file path')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--skip_vis', action='store_true')
    parser.add_argument('--skip_vis_step', action='store_true')
    parser.add_argument('--skip_vis_final', action='store_true')
    parser.add_argument('--skip_final', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.cfg is not None:
        cfg_data, cfg_exp = load_cfg_from_file(args.cfg, args)
    else:
        cfg_data, cfg_exp = load_cfg_from_cmd(args)

    if args.out is not None:
        cfg_exp.args.output = args.out
    out = cfg_exp.args.output
    os.makedirs(out, exist_ok=True)
    print(cfg_data, file=open(os.path.join(out, 'cfg_data.yml'), 'w'))
    print(cfg_exp, file=open(os.path.join(out, 'cfg_exp.yml'), 'w'))
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push(f"Load Dataset and Model")
    dataset = load_object(cfg_data.module, cfg_data.args)
    print(dataset)

    model = load_object(cfg_exp.module, cfg_exp.args)
    torch.cuda.nvtx.range_pop()
    elapsed_time1 = time.time() - start_time1

    
    # process(dataset, model, args)
    process_batched(dataset, model, args)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Configuration & initialization time: {elapsed_time1:.2f} seconds")
    print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")

if __name__ == '__main__':
    main_entrypoint()