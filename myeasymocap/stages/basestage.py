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
# import multiprocessing as mp
import os
from math import floor

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

    def at_step(self, data, index):
        ret = {}
        if 'meta' in data:
            ret['meta'] = data['meta']
        for key in self.keys_keep:
            ret[key] = data[key]

        timer = {}
        for key, model in self.model_steps.items():
            if self._at_step[key].get('skip', False):
                continue

            inputs = {}
            for k in self._at_step[key].get('key_from_data', []):
                inputs[k] = data[k]
            for k in self._at_step[key].get('key_from_previous', []):
                inputs[k] = ret[k]

            # ===== NVTX 範圍開始 =====
            torch.cuda.nvtx.range_push(f"{key}_frame{index}")
            # ========================
            

            start = time.time()
            output = model(**inputs)
            timer[key] = time.time() - start


            torch.cuda.nvtx.range_pop()

            if output is not None:
                ret.update(output)

        # 更新 Timer
        self.timer.update(timer)
        # 存每個 sample 的耗時
        self.last_step = timer.copy()

        return ret

    # def at_step(self, batch_data, batch_indices):
    #     batch_ret = []   
    #     batch_timer = [] 

    #     detect_key = 'detect'
    #     detect_time = 0.0  

    #     # --- Step 1: detect 批次處理 ---
    #     if detect_key in self.model_steps and not self._at_step[detect_key].get('skip', False):
    #         all_images, all_imgnames = [], []
    #         view_counts = []
    #         for data in batch_data:
    #             all_images.extend(data['images'])
    #             all_imgnames.extend(data['imgnames'])
    #             view_counts.append(len(data['images']))

    #         torch.cuda.nvtx.range_push(f"detect_{batch_indices}_batch")
    #         start = time.time()
    #         batch_detects = self.model_steps[detect_key](all_images, all_imgnames)
    #         detect_time = time.time() - start
    #         torch.cuda.nvtx.range_pop()

    #         # 拆回每個樣本
    #         start_idx = 0
    #         for i, data in enumerate(batch_data):
    #             n_views = view_counts[i]
    #             ret = {}
    #             if 'meta' in data:
    #                 ret['meta'] = data['meta']
    #             for key in self.keys_keep:
    #                 ret[key] = data[key]
    #             ret['bbox'] = batch_detects['bbox'][start_idx:start_idx+n_views]

    #             batch_ret.append(ret)
    #             batch_timer.append({})  # 不再重複加 detect_time
    #             start_idx += n_views
    #     else:
    #         # detect 被 skip
    #         for data, index in zip(batch_data, batch_indices):
    #             ret = {}
    #             if 'meta' in data:
    #                 ret['meta'] = data['meta']
    #             for key in self.keys_keep:
    #                 ret[key] = data[key]
    #             batch_ret.append(ret)
    #             batch_timer.append({})

    #     # --- Step 2: 其他 steps ---
    #     for key, model in self.model_steps.items():
    #         if key == detect_key or self._at_step[key].get('skip', False):
    #             continue

    #         # 特別處理 keypoints2d: 改為批次模式
    #         if key == 'keypoints2d':
    #             all_images, all_bboxes, all_imgnames = [], [], []
    #             view_counts = []
    #             for data, ret in zip(batch_data, batch_ret):
    #                 all_images.extend(data['images'])
    #                 all_bboxes.extend(ret['bbox'])
    #                 all_imgnames.extend(data['imgnames'])
    #                 view_counts.append(len(data['images']))

    #             start = time.time()
    #             torch.cuda.nvtx.range_push(f"keypoints2d_{batch_indices}_batch")
    #             batch_keypoints = model(all_bboxes, all_images, all_imgnames)
    #             torch.cuda.nvtx.range_pop()
    #             total_time = time.time() - start

    #             start_idx = 0
    #             for i, (data, ret) in enumerate(zip(batch_data, batch_ret)):
    #                 n_views = view_counts[i]
    #                 ret['keypoints'] = batch_keypoints['keypoints'][start_idx:start_idx+n_views]
    #                 batch_timer[i][key] = total_time / len(batch_data)
    #                 start_idx += n_views
    #             continue



    #         # --- 其他步驟維持逐樣本處理 ---
    #         else:
    #             for i, (data, index) in enumerate(zip(batch_data, batch_indices)):
    #                 ret = batch_ret[i]
    #                 timer = {}

    #                 # 準備 inputs
    #                 inputs = {}
    #                 for k in self._at_step[key].get('key_from_data', []):
    #                     inputs[k] = data[k]
    #                 for k in self._at_step[key].get('key_from_previous', []):
    #                     inputs[k] = ret[k]

    #                 start = time.time()
    #                 torch.cuda.nvtx.range_push(f"{key}_frame{index}")
    #                 output = model(**inputs)
    #                 torch.cuda.nvtx.range_pop()
    #                 timer[key] = time.time() - start

    #                 if output is not None:
    #                     ret.update(output)
    #                 batch_timer[i].update(timer)


    #     # === Step 3: 合併批次計時 ===
    #     merged_timer = {}
    #     for t in batch_timer:
    #         for k, v in t.items():
    #             merged_timer[k] = merged_timer.get(k, 0.0) + v


    #     if detect_time > 0:
    #         merged_timer[detect_key] = merged_timer.get(detect_key, 0.0) + detect_time


    #     self.last_step = merged_timer.copy()  
    #     self.timer.update(merged_timer)       

    #     return batch_ret


    



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

        # 初始化 timer
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


            # 輸出計時結果
            self.timer.update(timer_record)

            for key in self.keys_keep:
                result[key] = ret0[key]

        return {'results': results}

class StageForFittingEach_MT:
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

        # 初始化 timer
        self.timer = Timer(record={k: 0. for k in self.stages.keys()},
                           verbose=verbose)

    def __call__(self, results, **ret):

        results_items = list(results.items())
        timer_records = {}

        # 定義單個人物處理函數
        def process_one_person(pid_result_tuple):
            pid, result = pid_result_tuple
            print(f'[{self.__class__.__name__}] Optimize person {pid} with {len(result["frames"])} frames')

            ret0 = {}
            ret0.update(ret)
            timer_record = {}

            for key, stage in self.stages.items():
                start = time.time()
                # ===== NVTX 範圍開始 =====
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
                    if output is not None:
                        ret0.update(output)
                torch.cuda.nvtx.range_pop()
                # ===== NVTX 範圍結束 =====
                end = time.time()
                timer_record[key] = end - start

            # 更新結果
            for key in self.keys_keep:
                result[key] = ret0[key]

            return pid, timer_record, result

        with ThreadPoolExecutor(max_workers=len(results_items)) as executor:
            futures = [executor.submit(process_one_person, item) for item in results_items]

            for future in as_completed(futures):
                pid, timer_record, result = future.result()
                timer_records[pid] = timer_record
                results[pid] = result

        # 更新總計時器
        for pid, timer_record in timer_records.items():
            self.timer.update(timer_record)

        return {'results': results}

# class StageForFittingEach_Hogwild:
#     def __init__(self, stages, keys_keep, verbose=True):
#         stages_ = {}
#         for key, val in stages.items():
#             if val['module'] == 'skip':
#                 mywarn(f'Stage {key} is not used')
#                 continue
#             model = load_object(val['module'], val['args'])
#             stages_[key] = model
#         self.stages = stages_
#         self.stages_args = stages
#         self.keys_keep = keys_keep
#         self.timer = Timer(record={k: 0. for k in self.stages.keys()}, verbose=verbose)

#     # 這個 method 可以被 spawn pickle
#     def process_one_person(self, pid_result_tuple, ret_dict, timer_dict, ret, shared_params_dict):
#         pid, result = pid_result_tuple

#         # ⚠ spawn 啟動，初始化 CUDA
#         if torch.cuda.is_available():
#             torch.cuda.init()

#         print(f'[{self.__class__.__name__}] Optimize person {pid} with {len(result["frames"])} frames')

#         # 使用共享參數 tensor
#         ret0 = {}
#         ret0.update(ret)
#         timer_record = {}

#         # 把每個 params 放入 shared tensor
#         for key, param in result.get("params", {}).items():
#             if key not in shared_params_dict:
#                 t = torch.tensor(param, dtype=torch.float32, device="cuda", requires_grad=True)
#                 t.share_memory_()  # CPU shared memory 也可以，Hogwild 就能操作
#                 shared_params_dict[key] = t

#         for key, stage in self.stages.items():
#             start = time.time()
#             torch.cuda.nvtx.range_push(f"{key}_pid{pid}")

#             for iter_ in range(self.stages_args[key].get('repeat', 1)):
#                 inputs = {}
#                 stage.iter = iter_
#                 # key_from_data 取自 result
#                 for k in self.stages_args[key].get('key_from_data', []):
#                     if k == "params":
#                         inputs[k] = shared_params_dict
#                     else:
#                         inputs[k] = result.get(k)
#                 for k in self.stages_args[key].get('key_from_previous', []):
#                     inputs[k] = ret0.get(k)

#                 output = stage(**inputs)
#                 if output is not None:
#                     ret0.update(output)

#             torch.cuda.nvtx.range_pop()
#             end = time.time()
#             timer_record[key] = end - start

#         # 更新 keys_keep
#         for key in self.keys_keep:
#             result[key] = ret0.get(key, None)

#         ret_dict[pid] = result
#         timer_dict[pid] = timer_record

#     def __call__(self, results, **ret):
#         results_items = list(results.items())
#         timer_records = {}

#         mp.set_start_method("spawn", force=True)

#         with mp.Manager() as manager:
#             ret_dict = manager.dict()
#             timer_dict = manager.dict()
#             shared_params_dict = manager.dict()  # 多 process 共用 tensor

#             processes = []
#             for item in results_items:
#                 p = mp.Process(
#                     target=self.process_one_person,
#                     args=(item, ret_dict, timer_dict, ret, shared_params_dict)
#                 )
#                 p.start()
#                 processes.append(p)

#             for p in processes:
#                 p.join()

#             # 將結果回寫
#             for pid in ret_dict.keys():
#                 results[pid] = ret_dict[pid]
#                 timer_records[pid] = timer_dict[pid]

#         for pid, timer_record in timer_records.items():
#             self.timer.update(timer_record)

#         return {'results': results}
class StageForFittingEach_MP:
    def __init__(self, stages, keys_keep, num_processes=None, verbose=True) -> None:
        stages_ = {}
        for key, val in stages.items():
            if val['module'] == 'skip':
                print(f'Stage {key} is not used')
                continue
            model = load_object(val['module'], val['args'])
            stages_[key] = model
        
        self.stages = stages_
        self.stages_args = stages
        self.keys_keep = keys_keep
        self.verbose = verbose

        if num_processes is None:
            self.num_processes = min(mp.cpu_count(), 4)  # 預設最多4個進程
        else:
            self.num_processes = num_processes

        self.num_threads_per_process = floor(mp.cpu_count() / self.num_processes)

        self.timer = Timer(record={k: 0. for k in self.stages.keys()},
                          verbose=verbose)

    @staticmethod
    def fit_single_person(pid, result, stages, stages_args, keys_keep, num_threads, shared_ret, device_id=None):

        torch.set_num_threads(num_threads)

        if device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        
        print(f'[Process-{os.getpid()}] Optimize person {pid} with {len(result["frames"])} frames')
        
        ret0 = {}
        ret0.update(shared_ret)
        
        timer_record = {}
        
        for key, stage in stages.items():
            start = time.time()
            
            
            torch.cuda.nvtx.range_push(f"{key}_pid{pid}")
            
            for iter_ in range(stages_args[key].get('repeat', 1)):
                inputs = {}
                stage.iter = iter_
                
                for k in stages_args[key].get('key_from_data', []):
                    inputs[k] = result[k]
                
                for k in stages_args[key].get('key_from_previous', []):
                    inputs[k] = ret0[k]
                
                output = stage(**inputs)
                if output is not None:
                    ret0.update(output)
            
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
            
            end = time.time()
            timer_record[key] = end - start
        
        result_output = {key: ret0[key] for key in keys_keep}
        
        return pid, result_output, timer_record

    def __call__(self, results, **ret):
        pids = list(results.keys())
        num_people = len(pids)
        
        if num_people == 0:
            return {'results': results}
        
        actual_num_processes = min(self.num_processes, num_people)
        
        # print(f'[{self.__class__.__name__}] Processing {num_people} people with {actual_num_processes} processes')
        # print(f'[{self.__class__.__name__}] Each process uses {self.num_threads_per_process} threads')
        
        shared_ret = {k: v for k, v in ret.items()}
        
  
        mp_context = mp.get_context('spawn')
        

        with mp_context.Pool(processes=actual_num_processes) as pool:
            tasks = []
            for pid in pids:
                task = pool.apply_async(
                    self.fit_single_person,
                    args=(
                        pid,
                        results[pid],
                        self.stages,
                        self.stages_args,
                        self.keys_keep,
                        self.num_threads_per_process,
                        shared_ret,
                        None  
                    )
                )
                tasks.append(task)
            
            all_timer_records = {k: [] for k in self.stages.keys()}
            
            for task in tasks:
                pid, result_output, timer_record = task.get()

                results[pid].update(result_output)
                
                for key, duration in timer_record.items():
                    all_timer_records[key].append(duration)
        
        avg_timer_record = {
            k: sum(v) / len(v) if v else 0.0 
            for k, v in all_timer_records.items()
        }
        self.timer.update(avg_timer_record)
        
        if self.verbose:
            print(f'[{self.__class__.__name__}] All {num_people} people processed')
        
        return {'results': results}