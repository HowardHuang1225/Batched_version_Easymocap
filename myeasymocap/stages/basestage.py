from typing import Any
from easymocap.config import Config, load_object
from easymocap.mytools.debug_utils import mywarn, log
import numpy as np
import time
from tabulate import tabulate
import torch
# from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
# import os
import time
# import nvtx
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import torch.multiprocessing as mp
import os
from math import floor
from queue import Queue
import math
from multiprocessing import Pool
from functools import partial

class Timer:
    def __init__(self, record, verbose) -> None:
        self.keys = list(record.keys())
        self.header = self.keys
        self.verbose = verbose

    def update(self, timer):
        if not self.verbose:
            return
        contents = []
        for key in self.keys:
            if key not in timer:
                contents.append('skip')
            else:
                contents.append('{:.3f}s'.format(timer[key]))
        print(tabulate(headers=self.header, tabular_data=[contents], tablefmt='fancy_grid'))

class MultiStage:
    def load_final(self):
        at_finals = {}
        for key, val in self._at_final.items():
            if 'module' not in val.keys():
                continue
            if val['module'] == 'skip':
                mywarn('Stage {} is not used'.format(key))
                continue
            log('[{}] loading {}'.format(self.__class__.__name__, key))
            model = load_object(val['module'], val['args'])
            model.output = self.output
            at_finals[key] = model
        self.model_finals = at_finals

    def __init__(self, output, at_step, at_final, keys_keep=[], timer=True) -> None:
        log('[{}] writing the results to {}'.format(self.__class__.__name__, output))
        at_steps = {}
        for key, val in at_step.items():
            if val['module'] == 'skip':
                mywarn('Stage {} is not used'.format(key))
                continue
            log('[{}] loading module {}'.format(self.__class__.__name__, key))
            model = load_object(val['module'], val['args'])
            model.output = output
            at_steps[key] = model
        self.output = output
        self.model_steps = at_steps
        self._at_step = at_step
        self._at_final = at_final
        self.keys_keep = keys_keep
        self.timer = Timer(at_steps, verbose=timer)

    # def at_step(self, data, index):
    #     ret = {}
    #     if 'meta' in data:
    #         ret['meta'] = data['meta']
    #     for key in self.keys_keep:
    #         ret[key] = data[key]

    #     timer = {}
    #     for key, model in self.model_steps.items():
    #         if self._at_step[key].get('skip', False):
    #             continue

    #         inputs = {}
    #         for k in self._at_step[key].get('key_from_data', []):
    #             inputs[k] = data[k]
    #         for k in self._at_step[key].get('key_from_previous', []):
    #             inputs[k] = ret[k]

    #         # ===== NVTX 範圍開始 =====
    #         # if key in ['detect', 'keypoints2d']:
    #         #     torch.cuda.nvtx.range_push(f"{key}_frame{index}")
    #         # ========================
            

    #         start = time.time()
    #         output = model(**inputs)
    #         timer[key] = time.time() - start

    #         # if key in ['detect', 'keypoints2d']:
    #         #     torch.cuda.nvtx.range_pop()

    #         if output is not None:
    #             ret.update(output)

    #     # 更新 Timer
    #     self.timer.update(timer)
    #     # 存每個 sample 的耗時
    #     self.last_step = timer.copy()

    #     return ret

    def at_step(self, batch_data, batch_indices):
        batch_ret = []   
        batch_timer = [] 

        detect_key = 'detect'
        detect_time = 0.0  

        # --- Step 1: detect 批次處理 ---
        if detect_key in self.model_steps and not self._at_step[detect_key].get('skip', False):
            all_images, all_imgnames = [], []
            view_counts = []
            for data in batch_data:
                all_images.extend(data['images'])
                all_imgnames.extend(data['imgnames'])
                view_counts.append(len(data['images']))

            torch.cuda.nvtx.range_push(f"detect_{batch_indices}_batch")
            start = time.time()
            batch_detects = self.model_steps[detect_key](all_images, all_imgnames)
            detect_time = time.time() - start
            torch.cuda.nvtx.range_pop()

            # 拆回每個樣本
            start_idx = 0
            for i, data in enumerate(batch_data):
                n_views = view_counts[i]
                ret = {}
                if 'meta' in data:
                    ret['meta'] = data['meta']
                for key in self.keys_keep:
                    ret[key] = data[key]
                ret['bbox'] = batch_detects['bbox'][start_idx:start_idx+n_views]

                batch_ret.append(ret)
                batch_timer.append({})  # 不再重複加 detect_time
                start_idx += n_views
        else:
            # detect 被 skip
            for data, index in zip(batch_data, batch_indices):
                ret = {}
                if 'meta' in data:
                    ret['meta'] = data['meta']
                for key in self.keys_keep:
                    ret[key] = data[key]
                batch_ret.append(ret)
                batch_timer.append({})

        # --- Step 2: 其他 steps ---
        for key, model in self.model_steps.items():
            if key == detect_key or self._at_step[key].get('skip', False):
                continue

            # 特別處理 keypoints2d: 改為批次模式
            if key == 'keypoints2d':
                all_images, all_bboxes, all_imgnames = [], [], []
                view_counts = []
                for data, ret in zip(batch_data, batch_ret):
                    all_images.extend(data['images'])
                    all_bboxes.extend(ret['bbox'])
                    all_imgnames.extend(data['imgnames'])
                    view_counts.append(len(data['images']))

                start = time.time()
                torch.cuda.nvtx.range_push(f"keypoints2d_{batch_indices}_batch")
                batch_keypoints = model(all_bboxes, all_images, all_imgnames)
                torch.cuda.nvtx.range_pop()
                total_time = time.time() - start

                start_idx = 0
                for i, (data, ret) in enumerate(zip(batch_data, batch_ret)):
                    n_views = view_counts[i]
                    ret['keypoints'] = batch_keypoints['keypoints'][start_idx:start_idx+n_views]
                    batch_timer[i][key] = total_time / len(batch_data)
                    start_idx += n_views
                continue



            # --- 其他步驟維持逐樣本處理 ---
            else:
                for i, (data, index) in enumerate(zip(batch_data, batch_indices)):
                    ret = batch_ret[i]
                    timer = {}

                    # 準備 inputs
                    inputs = {}
                    for k in self._at_step[key].get('key_from_data', []):
                        inputs[k] = data[k]
                    for k in self._at_step[key].get('key_from_previous', []):
                        inputs[k] = ret[k]

                    start = time.time()
                    torch.cuda.nvtx.range_push(f"{key}_frame{index}")
                    output = model(**inputs)
                    torch.cuda.nvtx.range_pop()
                    timer[key] = time.time() - start

                    if output is not None:
                        ret.update(output)
                    batch_timer[i].update(timer)


        # === Step 3: 合併批次計時 ===
        merged_timer = {}
        for t in batch_timer:
            for k, v in t.items():
                merged_timer[k] = merged_timer.get(k, 0.0) + v


        if detect_time > 0:
            merged_timer[detect_key] = merged_timer.get(detect_key, 0.0) + detect_time


        self.last_step = merged_timer.copy()  
        self.timer.update(merged_timer)       

        return batch_ret


    



    @staticmethod
    def merge_data(infos_all):
        info0 = infos_all[0]
        data = {}
        for key, val in info0.items():
            data[key] = [info[key] for info in infos_all]
            if isinstance(val, np.ndarray):
                try:
                    data[key] = np.stack(data[key])
                except ValueError:
                    print('[{}] Skip merge {}'.format('Stages', key))
                    pass
            elif isinstance(val, dict):
                data[key] = MultiStage.merge_data(data[key])
        return data

    # def at_final(self, infos_all):
    #     self.load_final()
    #     data = self.merge_data(infos_all)
    #     log('Keep keys: {}'.format(list(data.keys())))
    #     ret = {}
    #     for key, model in self.model_finals.items():
    #         if self._at_final[key].get('skip', False):
    #             continue
    #         for iter_ in range(self._at_final[key].get('repeat', 1)):
    #             inputs = {}
    #             model.iter = iter_
    #             for k in self._at_final[key].get('key_from_data', []):
    #                 inputs[k] = data[k]
    #             for k in self._at_final[key].get('key_from_previous', []):
    #                 inputs[k] = ret[k]
    #             try:
    #                 output = model(**inputs)
    #             except:
    #                 print('[{}] Error in {}'.format('Stages', key))
    #                 raise Exception
    #             if output is not None:
    #                 ret.update(output)
    #     return ret
    def at_final(self, infos_all):
        start_final_total = time.time()  # 整個 at_final 的 wall-clock
        ret = {}
        final_timer = {}  # 記錄每個子模組及 preprocess 的耗時

        # ==== Preprocess 計時 ====
        start = time.time()
        torch.cuda.nvtx.range_push(f"final_data_preprocessing")
        self.load_final()
        data = self.merge_data(infos_all)
        torch.cuda.nvtx.range_pop()
        final_timer['data preprocessing'] = time.time() - start

        log('Keep keys: {}'.format(list(data.keys())))

        # ==== 各子模組計時 ====
        for key, model in self.model_finals.items():
            if self._at_final[key].get('skip', False):
                continue
            for iter_ in range(self._at_final[key].get('repeat', 1)):
                inputs = {k: data[k] for k in self._at_final[key].get('key_from_data', [])}
                inputs.update({k: ret[k] for k in self._at_final[key].get('key_from_previous', [])})

                model.iter = iter_

                start = time.time()
                torch.cuda.nvtx.range_push(f"{key}_final")
                if key == 'fitting_each_person':
                    mp.set_start_method('spawn', force=True)
                output = model(**inputs)
                torch.cuda.nvtx.range_pop()
                elapsed = time.time() - start

                # 累加到 final_timer
                final_timer[key] = final_timer.get(key, 0.0) + elapsed

                if output is not None:
                    ret.update(output)

        self.last_final = final_timer.copy()
        self.timer.update(final_timer)

        return ret




# class StageForFittingEach:
#     def __init__(self, stages, keys_keep) -> None:
#         stages_ = {}
#         for key, val in stages.items():
#             if val['module'] == 'skip':
#                 mywarn('Stage {} is not used'.format(key))
#                 continue
#             model = load_object(val['module'], val['args'])
#             stages_[key] = model
#         self.stages = stages_
#         self.stages_args = stages
#         self.keys_keep = keys_keep
    
#     def __call__(self, results, **ret):
#         for pid, result in results.items():
#             print('[{}] Optimize person {} with {} frames'.format(self.__class__.__name__, pid, len(result['frames'])))
#             ret0 = {}
#             ret0.update(ret)
#             for key, stage in self.stages.items():
#                 for iter_ in range(self.stages_args[key].get('repeat', 1)):
#                     inputs = {}
#                     stage.iter = iter_
#                     for k in self.stages_args[key].get('key_from_data', []):
#                         inputs[k] = result[k]
#                     for k in self.stages_args[key].get('key_from_previous', []):
#                         inputs[k] = ret0[k]
#                     output = stage(**inputs)
#                     if output is not None:
#                         ret0.update(output)
#             for key in self.keys_keep:
#                 result[key] = ret0[key]
#         return {'results': results}
class StageForFittingEach:
    def __init__(self, stages, keys_keep, verbose=True) -> None:
        stages_ = {}
        for key, val in stages.items():
            if val['module'] == 'skip':
                mywarn('Stage {} is not used'.format(key))
                continue
            model = load_object(val['module'], val['args'])
            stages_[key] = model
        self.stages = stages_
        self.stages_args = stages
        self.keys_keep = keys_keep

        self.timer = Timer(record={k: 0. for k in self.stages.keys()},
                           verbose=verbose)

    def __call__(self, results, **ret):

        for pid, result in results.items():
            print('[{}] Optimize person {} with {} frames'.format(
                self.__class__.__name__, pid, len(result['frames'])))
            ret0 = {}
            ret0.update(ret)
            
            timer_record = {}

            for key, stage in self.stages.items():


                start = time.time()
                # # ===== NVTX 範圍開始 =====
                torch.cuda.nvtx.range_push(f"{key}_pid{pid}")
                # ========================
                for iter_ in range(self.stages_args[key].get('repeat', 1)):
                    inputs = {}
                    stage.iter = iter_
                    for k in self.stages_args[key].get('key_from_data', []):
                        inputs[k] = result[k]
                    for k in self.stages_args[key].get('key_from_previous', []):
                        inputs[k] = ret0[k]
                    
                    output = stage(**inputs)
                    # print(inputs)
                    if output is not None:
                        ret0.update(output)
                torch.cuda.nvtx.range_pop()
                # ===== NVTX 範圍結束 =====
                end = time.time()
                timer_record[key] = end - start

            self.timer.update(timer_record)

            for key in self.keys_keep:
                result[key] = ret0[key]

        return {'results': results}
    

class StageForFittingEach_MT:
    def __init__(self, stages, keys_keep, verbose=True, max_workers=None) -> None:
        stages_ = {}
        for key, val in stages.items():
            if val['module'] == 'skip':
                mywarn(f'Stage {key} is not used')
                continue
            model = load_object(val['module'], val['args'])
            stages_[key] = model
        self.stages = stages_
        self.stages_args = stages
        self.keys_keep = keys_keep

        self.timer = Timer(record={k: 0. for k in self.stages.keys()},
                           verbose=verbose)

        # 限制最大同時執行人物數，避免 GPU context thrashing
        self.max_workers = max_workers or min(4, mp.cpu_count() // 2)

    def __call__(self, results, **ret):
        results_items = list(results.items())
        timer_records = {}

        def process_one_person(pid_result_tuple):
            pid, result = pid_result_tuple
            print(f'[{self.__class__.__name__}] Optimize person {pid} with {len(result["frames"])} frames')

            # 每個人物使用獨立 CUDA stream
            stream = torch.cuda.Stream()
            ret0 = dict(ret)
            timer_record = {}

            # 在 stream 上運行 GPU 任務
            with torch.cuda.stream(stream):
                for key, stage in self.stages.items():
                    start = time.time()
                    torch.cuda.nvtx.range_push(f"{key}_pid{pid}")

                    for iter_ in range(self.stages_args[key].get('repeat', 1)):
                        inputs = {}
                        stage.iter = iter_
                        for k in self.stages_args[key].get('key_from_data', []):
                            inputs[k] = result[k]
                        for k in self.stages_args[key].get('key_from_previous', []):
                            inputs[k] = ret0[k]

                        output = stage(**inputs)
                        if output is not None:
                            ret0.update(output)

                    torch.cuda.nvtx.range_pop()
                    timer_record[key] = time.time() - start

            for key in self.keys_keep:
                result[key] = ret0[key]

            return pid, timer_record, result


        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(results_items))) as executor:
            futures = [executor.submit(process_one_person, item) for item in results_items]

            for future in as_completed(futures):
                pid, timer_record, result = future.result()
                timer_records[pid] = timer_record
                results[pid] = result

        # 所有 stream 都提交完後再同步 GPU
        torch.cuda.synchronize()

        for pid, timer_record in timer_records.items():
            self.timer.update(timer_record)

        return {'results': results}

# class StageForFittingEach_MT:
#     def __init__(self, stages, keys_keep, verbose=True) -> None:
#         stages_ = {}
#         for key, val in stages.items():
#             if val['module'] == 'skip':
#                 mywarn('Stage {} is not used'.format(key))
#                 continue
#             model = load_object(val['module'], val['args'])
#             stages_[key] = model
#         self.stages = stages_
#         self.stages_args = stages
#         self.keys_keep = keys_keep

#         # 初始化 timer
#         self.timer = Timer(record={k: 0. for k in self.stages.keys()},
#                            verbose=verbose)

#     def __call__(self, results, **ret):

#         results_items = list(results.items())
#         timer_records = {}

#         # 定義單個人物處理函數
#         def process_one_person(pid_result_tuple):
#             pid, result = pid_result_tuple
#             print(f'[{self.__class__.__name__}] Optimize person {pid} with {len(result["frames"])} frames')

#             ret0 = {}
#             ret0.update(ret)
#             timer_record = {}

#             for key, stage in self.stages.items():
#                 start = time.time()
#                 # ===== NVTX 範圍開始 =====
#                 torch.cuda.nvtx.range_push(f"{key}_pid{pid}")
#                 # ========================
#                 for iter_ in range(self.stages_args[key].get('repeat', 1)):
#                     inputs = {}
#                     stage.iter = iter_
#                     for k in self.stages_args[key].get('key_from_data', []):
#                         inputs[k] = result[k]
#                     for k in self.stages_args[key].get('key_from_previous', []):
#                         inputs[k] = ret0[k]

#                     output = stage(**inputs)
#                     if output is not None:
#                         ret0.update(output)
#                 torch.cuda.nvtx.range_pop()
#                 # ===== NVTX 範圍結束 =====
#                 end = time.time()
#                 timer_record[key] = end - start

#             # 更新結果
#             for key in self.keys_keep:
#                 result[key] = ret0[key]

#             return pid, timer_record, result

#         with ThreadPoolExecutor(max_workers=len(results_items)) as executor:
#             futures = [executor.submit(process_one_person, item) for item in results_items]

#             for future in as_completed(futures):
#                 pid, timer_record, result = future.result()
#                 timer_records[pid] = timer_record
#                 results[pid] = result

#         # 更新總計時器
#         for pid, timer_record in timer_records.items():
#             self.timer.update(timer_record)

#         return {'results': results}


# =============================
# initializer for worker processes
# =============================
_global_stages = None
_global_stages_args = None
_global_keys_keep = None

def _init_worker(stages_args, keys_keep):
    """子process初始化時預先載入模型"""
    global _global_stages, _global_stages_args, _global_keys_keep
    torch.cuda.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _global_stages_args = stages_args
    _global_keys_keep = keys_keep
    _global_stages = {}
    for key, val in stages_args.items():
        if val["module"] != "skip":
            model = load_object(val["module"], val["args"])
            if hasattr(model, "to") and callable(model.to):
                model.to(device)
            _global_stages[key] = model
    # print(f"[Worker {os.getpid()}] Initialized with {len(_global_stages)} stages on {device}")


# =============================
# Worker function
# =============================
def _process_one_person_worker(pid_result_tuple, ret):
    pid, result = pid_result_tuple
    global _global_stages, _global_stages_args, _global_keys_keep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(f"[Worker {os.getpid()}] Optimize person {pid} with {len(result['frames'])} frames")

    ret0 = dict(ret)
    timer_record = {}

    for key, stage in _global_stages.items():
        start = time.time()
        for iter_ in range(_global_stages_args[key].get("repeat", 1)):
            inputs = {}
            stage.iter = iter_

            for k in _global_stages_args[key].get("key_from_data", []):
                v = result[k]
                if isinstance(v, torch.Tensor):
                    v = v.pin_memory().to(device, non_blocking=True)
                inputs[k] = v

            for k in _global_stages_args[key].get("key_from_previous", []):
                v = ret0[k]
                if isinstance(v, torch.Tensor):
                    v = v.pin_memory().to(device, non_blocking=True)
                inputs[k] = v

            with torch.amp.autocast(device_type='cuda'):
                output = stage(**inputs)

            if output is not None:
                ret0.update(output)

        torch.cuda.synchronize()
        timer_record[key] = time.time() - start

    for key in _global_keys_keep:
        result[key] = ret0[key]

    return pid, timer_record, result


# =============================
# main part
# =============================
class StageForFittingEach_MP:
    def __init__(self, stages, keys_keep, verbose=True, num_workers=4):
        self.stages_args = stages
        self.keys_keep = keys_keep
        self.verbose = verbose

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() // 2)
        self.num_workers = num_workers

        self.timer = Timer(record={k: 0. for k in stages.keys()},
                           verbose=verbose)

        # 初始化 persistent pool
        mp.set_start_method("spawn", force=True)
        self.pool = Pool(processes=self.num_workers,
                         initializer=_init_worker,
                         initargs=(self.stages_args, self.keys_keep))

        print(f"[StageForFittingEach_MP_Persistent] Pool created with {self.num_workers} workers")

    def __call__(self, results, **ret):
        results_items = list(results.items())
        timer_records = {}

        # 平行處理
        worker_func = partial(_process_one_person_worker, ret=ret)
        outputs = self.pool.map(worker_func, results_items)

        for pid, timer_record, result in outputs:
            timer_records[pid] = timer_record
            results[pid] = result

        for pid, timer_record in timer_records.items():
            self.timer.update(timer_record)

        return {'results': results}

    def close(self):
        """釋放 pool"""
        print("[StageForFittingEach_MP_Persistent] Closing pool...")
        self.pool.close()
        self.pool.join()