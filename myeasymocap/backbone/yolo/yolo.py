import torch
import numpy as np
import os
import cv2
from os.path import join
import pickle
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

def check_modelpath(paths):
    if isinstance(paths, str):
        assert os.path.exists(paths), paths
        return paths
    elif isinstance(paths, list):
        for path in paths:
            if os.path.exists(path):
                print(f'Found model in {path}')
                break
        else:
            print(f'No model found in {paths}!')
            raise FileExistsError
        return path
    else:
        raise NotImplementedError



# --------------------------
# YOLOv11 Wrapper
# --------------------------
# class BaseYOLOv11:
#     def __init__(self, ckpt=None, model='yolov8m.engine', name='object2d', multiview=True) -> None:
#         if ckpt is not None:
#             ckpt = check_modelpath(ckpt)
#             self.model = YOLO(ckpt, task='detect')
#         else:
#             print(f'[{self.__class__.__name__}] Not given ckpt, use default {model}')
#             self.model = YOLO(model, task='detect')
#         self.multiview = multiview
#         self.name = name
#         self.output = 'output'

#     def dump(self, cachename, output):
#         os.makedirs(os.path.dirname(cachename), exist_ok=True)
#         with open(cachename, 'wb') as f:
#             pickle.dump(output, f)
#         return output

#     def load(self, cachename):
#         with open(cachename, 'rb') as f:
#             output = pickle.load(f)
#         return output

#     def check_cache(self, imgname):
#         basename = os.path.basename(imgname)
#         imgext = '.' + basename.split('.')[-1]
#         nv = imgname.split(os.sep)[-2]
#         cachename = join(self.output, self.name, nv, basename.replace(imgext, '.npy'))
#         os.makedirs(os.path.dirname(cachename), exist_ok=True)
#         if os.path.exists(cachename):
#             output = self.load(cachename)
#             return True, output, cachename
#         else:
#             return False, None, cachename

#     @staticmethod
#     def _read_image(x):
#         """讀圖，支援 str 或 np.array"""
#         return cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) if isinstance(x, str) else x

#     @torch.no_grad()
#     def detect_batch(self, images, imgnames):
#         results_to_return = []
#         images_to_process = []
#         imgnames_to_process = []
#         caches_to_save = []

#         # 先檢查 cache
#         for img, name in zip(images, imgnames):
#             flag, cache, cachename = self.check_cache(name)
#             if flag:
#                 results_to_return.append(cache)
#             else:
#                 images_to_process.append(img)  
#                 imgnames_to_process.append(name)
#                 caches_to_save.append(cachename)

#         if len(images_to_process) > 0:
#             with ThreadPoolExecutor() as executor:
#                 images_to_process = list(executor.map(self._read_image, images_to_process))

#             # TensorRT engine 可一次 batch 多張圖片
#             results_list = self.model(images_to_process)  # list[Results]

#             for res, cachename, img in zip(results_list, caches_to_save, images_to_process):
#                 boxes = res.boxes.data.cpu().numpy()
#                 r = {
#                     'results': boxes,
#                     'image_shape': img.shape,
#                     'names': self.model.names
#                 }
#                 self.dump(cachename, r)
#                 results_to_return.append(r)

#         return results_to_return


#     @staticmethod
#     def select_class(results, name):
#         select = []
#         for res in results['results']:
#             cls_id = int(res[5])
#             classname = results['names'][cls_id]
#             if classname != name:
#                 continue
#             select.append(res[:5])  # xyxy + conf only
#         if len(select) == 0:
#             return np.zeros((0,5), dtype=np.float32), results
#         select = np.stack(select).astype(np.float32)
#         return select, results

#     def select_bbox(self, select, results, imgname):
#         if select.shape[0] == 0:
#             return select
#         idx = np.argsort(select[:, -1])[::-1]
#         return select[idx[0:1]]

#     def __call__(self, images, imgnames):
#         """
#         images: list of np.array 或 str
#         imgnames: list of str
#         """
#         squeeze = False
#         if not isinstance(images, list):
#             images = [images]
#             imgnames = [imgnames]
#             squeeze = True

#         # 一次 batch 推理
#         batch_results = self.detect_batch(images, imgnames)

#         detects = {'bbox': [[] for _ in range(len(images))]}
#         for nv, res in enumerate(batch_results):
#             select, res = self.select_class(res, self.name)
#             select = self.select_bbox(select, res, imgnames[nv])
#             detects['bbox'][nv] = select

#         if squeeze:
#             detects['bbox'] = detects['bbox'][0]
#         return detects

class BaseYOLOv11:
    def __init__(self, ckpt=None, model='yolov8m.pt', name='object2d', multiview=True) -> None:
        if ckpt is not None:
            ckpt = check_modelpath(ckpt)
            self.model = YOLO(ckpt, task='detect')
        else:
            print(f'[{self.__class__.__name__}] Not given ckpt, use default {model}')
            self.model = YOLO(model, task='detect')
        self.multiview = multiview
        self.name = name
        self.output = 'output'

    def dump(self, cachename, output):
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        with open(cachename, 'wb') as f:
            pickle.dump(output, f)
        return output

    def load(self, cachename):
        with open(cachename, 'rb') as f:
            output = pickle.load(f)
        return output

    def check_cache(self, imgname):
        basename = os.path.basename(imgname)
        imgext = '.' + basename.split('.')[-1]
        nv = imgname.split(os.sep)[-2]
        cachename = join(self.output, self.name, nv, basename.replace(imgext, '.npy'))
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        if os.path.exists(cachename):
            output = self.load(cachename)
            return True, output, cachename
        else:
            return False, None, cachename

    @staticmethod
    def _read_image(x):
        return cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) if isinstance(x, str) else x

    @torch.no_grad()
    def detect_batch(self, images, imgnames):
        results_to_return = []
        images_to_process = []
        imgnames_to_process = []
        caches_to_save = []

        # 先檢查 cache
        for img, name in zip(images, imgnames):
            flag, cache, cachename = self.check_cache(name)
            if flag:
                results_to_return.append(cache)
            else:
                images_to_process.append(img) 
                imgnames_to_process.append(name)
                caches_to_save.append(cachename)

        if len(images_to_process) > 0:
            with ThreadPoolExecutor() as executor:
                images_to_process = list(executor.map(self._read_image, images_to_process))

            results_list = self.model(images_to_process)  # list[Results]

            for res, cachename, img in zip(results_list, caches_to_save, images_to_process):
                boxes = res.boxes.data.cpu().numpy()
                r = {
                    'results': boxes,
                    'image_shape': img.shape,
                    'names': self.model.names
                }
                self.dump(cachename, r)
                results_to_return.append(r)

        return results_to_return


    @staticmethod
    def select_class(results, name):
        select = []
        for res in results['results']:
            cls_id = int(res[5])
            classname = results['names'][cls_id]
            if classname != name:
                continue
            select.append(res[:5])  # xyxy + conf only
        if len(select) == 0:
            return np.zeros((0,5), dtype=np.float32), results
        select = np.stack(select).astype(np.float32)
        return select, results

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        idx = np.argsort(select[:, -1])[::-1]
        return select[idx[0:1]]

    def __call__(self, images, imgnames):
        """
        images: list of np.array 或 str
        imgnames: list of str
        """
        squeeze = False
        if not isinstance(images, list):
            images = [images]
            imgnames = [imgnames]
            squeeze = True

        # 一次 batch 推理
        batch_results = self.detect_batch(images, imgnames)

        detects = {'bbox': [[] for _ in range(len(images))]}
        for nv, res in enumerate(batch_results):
            select, res = self.select_class(res, self.name)
            select = self.select_bbox(select, res, imgnames[nv])
            detects['bbox'][nv] = select

        if squeeze:
            detects['bbox'] = detects['bbox'][0]
        return detects




    
# class BaseYOLOv5:
#     def __init__(self, ckpt=None, model='yolov5m', name='object2d', multiview=True) -> None:
#         if ckpt is not None:
#             ckpt = check_modelpath(ckpt)
#             self.model = torch.hub.load('ultralytics/yolov5', 'custom', ckpt)
#         else:
#             print('[{}] Not given ckpt, use default yolov5'.format(self.__class__.__name__))
#             self.model = torch.hub.load('ultralytics/yolov5', model)
#         # self.model = torch.compile(self.model)
#         self.multiview = multiview
#         self.name = name
#         self.output = 'output'
    
#     def dump(self, cachename, output):
#         os.makedirs(os.path.dirname(cachename), exist_ok=True)
#         with open(cachename, 'wb') as f:
#             pickle.dump(output, f)
#         return output
    
#     def load(self, cachename):
#         with open(cachename, 'rb') as f:
#             output = pickle.load(f)
#         return output

#     def check_cache(self, imgname):
#         basename = os.path.basename(imgname)
#         imgext = '.' + basename.split('.')[-1]
#         nv = imgname.split(os.sep)[-2]
#         cachename = join(self.output, self.name, nv, basename.replace(imgext, '.npy'))
#         os.makedirs(os.path.dirname(cachename), exist_ok=True)
#         if os.path.exists(cachename):
#             output = self.load(cachename)
#             return True, output, cachename
#         else:
#             return False, None, cachename
    
#     def check_image(self, img_or_name):
#         if isinstance(img_or_name, str):
#             images = cv2.imread(img_or_name)
#         else:
#             images = img_or_name
#         images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
#         return images
    
#     @torch.no_grad()
#     def detect(self, image, imgname):
#         flag, cache, cachename = self.check_cache(imgname)
#         if flag:
#             return cache
#         image = self.check_image(imgname)
#         results = self.model(image) #RGB images[:,:,::-1]
#         arrays = np.array(results.pandas().xyxy[0])
#         res = {
#             'results': arrays,
#             'image_shape': image.shape,
#         }
#         self.dump(cachename, res)
#         return res
    
#     @staticmethod
#     def select_class(results, name):
#         select = []
#         for i, res in enumerate(results['results']):
#             classname = res[6]
#             if classname != name:
#                 continue
#             box = res[:5]
#             select.append(box)
#         select = np.stack(select)
#         return select, results


#     def select_bbox(self, select, results, imgname):
#         if select.shape[0] == 0:
#             return select
#         # Naive: select the best
#         idx = np.argsort(select[:, -1])[::-1]
#         return select[idx[0:1]]
#     # def select_bbox(self, select, top_k=1):
#     #     if select.shape[0] == 0:
#     #         return select
#     #     idx = np.argsort(select[:, 4])[::-1]
#     #     return select[idx[:top_k]]


#     def __call__(self, images, imgnames):
#         squeeze = False
#         if not isinstance(images, list):
#             images = [images]
#             imgnames = [imgnames]
#             squeeze = True

#         detects = {'bbox': [[] for _ in range(len(images))]}

#         # --------------------------
#         # Step 1: 檢查 cache
#         # --------------------------
#         cache_flags, cache_results, cache_names, batch_imgs = [], [], [], []
#         for img, name in zip(images, imgnames):
#             flag, cache, cachename = self.check_cache(name)
#             cache_flags.append(flag)
#             cache_results.append(cache)
#             cache_names.append(cachename)
#             batch_imgs.append(None)

#         # --------------------------
#         # Step 2: 批量推論未 cache 的影像
#         # --------------------------
#         need_predict_idx = [i for i, f in enumerate(cache_flags) if not f]
#         if len(need_predict_idx) > 0:
#             # input_batch = [batch_imgs[i] for i in need_predict_idx]
#             input_batch = [self.check_image(images[i]) for i in need_predict_idx]
#             print(f"[YOLOv5] Batch predict {len(input_batch)} images")
#             results_batch = self.model(input_batch)  # 一次處理多張
        
#             for idx, i in enumerate(need_predict_idx):
#                 arrays = np.array(results_batch.pandas().xyxy[idx])
#                 res = {'results': arrays, 'image_shape': input_batch[idx].shape}
#                 self.dump(cache_names[i], res)
#                 cache_results[i] = res

#         # --------------------------
#         # Step 3: 後處理每張影像
#         # --------------------------
#         for nv in range(len(images)):
#             res = cache_results[nv]
#             select, _ = self.select_class(res, self.name)
#             if len(select) == 0:
#                 select = np.zeros((0,5), dtype=np.float32)
#             else:
#                 select = np.array(select).astype(np.float32)
#                 select = self.select_bbox(select, res, imgnames[nv])

#             detects['bbox'][nv] = select

#         if squeeze:
#             detects['bbox'] = detects['bbox'][0]
#         return detects

#     # def __call__(self, images, imgnames): # 这里好像默认是多视角了，需要继承一下单视角的
#     #     squeeze = False
#     #     if not isinstance(images, list):
#     #         images = [images]
#     #         imgnames = [imgnames]
#     #         squeeze = True
#     #     detects = {'bbox': [[] for _ in range(len(images))]}
#     #     for nv in range(len(images)):
#     #         res = self.detect(images[nv], imgnames[nv])            
#     #         select, res = self.select_class(res, self.name)
#     #         if len(select) == 0:
#     #             select = np.zeros((0,5), dtype=np.float32)
#     #         else:
#     #             select = np.stack(select).astype(np.float32)
#     #         # TODO: add track here
#     #         select = self.select_bbox(select, res, imgnames[nv])
#     #         detects['bbox'][nv] = select
#     #     if squeeze:
#     #         detects['bbox'] = detects['bbox'][0]
#     #     return detects

class BaseYOLOv5:
    def __init__(self, ckpt=None, model='yolov5m', name='object2d', multiview=True) -> None:
        if ckpt is not None:
            ckpt = check_modelpath(ckpt)
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', ckpt)
        else:
            print('[{}] Not given ckpt, use default yolov5'.format(self.__class__.__name__))
            self.model = torch.hub.load('ultralytics/yolov5', model)
        # self.model = torch.compile(self.model)
        self.multiview = multiview
        self.name = name
        self.output = 'output'
    
    def dump(self, cachename, output):
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        with open(cachename, 'wb') as f:
            pickle.dump(output, f)
        return output
    
    def load(self, cachename):
        with open(cachename, 'rb') as f:
            output = pickle.load(f)
        return output

    def check_cache(self, imgname):
        basename = os.path.basename(imgname)
        imgext = '.' + basename.split('.')[-1]
        nv = imgname.split(os.sep)[-2]
        cachename = join(self.output, self.name, nv, basename.replace(imgext, '.npy'))
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        if os.path.exists(cachename):
            output = self.load(cachename)
            return True, output, cachename
        else:
            return False, None, cachename
    
    def check_image(self, img_or_name):
        """單張影像讀取與轉換"""
        if isinstance(img_or_name, str):
            images = cv2.imread(img_or_name, cv2.IMREAD_COLOR)
            if images is None:
                raise ValueError(f"Failed to read image: {img_or_name}")
        else:
            images = img_or_name
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        return images


    def check_images_batch(self, img_or_names, max_workers=32):
        """批次並行讀取影像"""
        from concurrent.futures import ThreadPoolExecutor
        
        def load_one(img_or_name):
            if isinstance(img_or_name, str):
                img = cv2.imread(img_or_name, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to read image: {img_or_name}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # 如果已經是 numpy array
                if isinstance(img_or_name, np.ndarray):
                    # 假設輸入可能是 BGR，需要轉換
                    if len(img_or_name.shape) == 3 and img_or_name.shape[2] == 3:
                        return cv2.cvtColor(img_or_name, cv2.COLOR_BGR2RGB)
                return img_or_name
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            images = list(executor.map(load_one, img_or_names))
        
        return images
    
    @torch.no_grad()
    def detect(self, image, imgname):
        flag, cache, cachename = self.check_cache(imgname)
        if flag:
            return cache
        image = self.check_image(imgname)
        results = self.model(image) #RGB images[:,:,::-1]
        arrays = np.array(results.pandas().xyxy[0])
        res = {
            'results': arrays,
            'image_shape': image.shape,
        }
        self.dump(cachename, res)
        return res
    
    @staticmethod
    def select_class(results, name):
        select = []
        for i, res in enumerate(results['results']):
            classname = res[6]
            if classname != name:
                continue
            box = res[:5]
            select.append(box)
        select = np.stack(select)
        return select, results


    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        # Naive: select the best
        idx = np.argsort(select[:, -1])[::-1]
        return select[idx[0:1]]

    @torch.no_grad()
    def detect_batch(self, images, parallel_load=True):
        """批次推論（支援並行讀取）"""
        if parallel_load:
            input_batch = self.check_images_batch(images)
        else:
            input_batch = [self.check_image(img) for img in images]
        
        print(f"[YOLOv5] Batch predict {len(input_batch)} images")
        
        results_batch = self.model(input_batch)
        
        pandas_results = results_batch.pandas().xyxy
        
        results = []
        for idx in range(len(images)):
            arrays = np.array(pandas_results[idx])
            res = {
                'results': arrays, 
                'image_shape': input_batch[idx].shape
            }
            results.append(res)
        
        return results


    def __call__(self, images, imgnames=None):
        squeeze = False
        if not isinstance(images, list):
            images = [images]
            squeeze = True
        
        num_images = len(images)
        detects = {'bbox': [None] * num_images}
        
        results = self.detect_batch(images, parallel_load=True)

        for i in range(num_images):
            res = results[i]
            results_array = res['results']
            
            if len(results_array) == 0:
                detects['bbox'][i] = np.zeros((0, 5), dtype=np.float32)
                continue
            
            class_mask = results_array[:, 6] == self.name
            select = results_array[class_mask, :5].astype(np.float32)
            
            if len(select) == 0:
                detects['bbox'][i] = np.zeros((0, 5), dtype=np.float32)
            else:
                imgname = imgnames[i] if imgnames is not None else None
                detects['bbox'][i] = self.select_bbox(select, res, imgname)
        
        if squeeze:
            detects['bbox'] = detects['bbox'][0]
        
        return detects


class BaseYOLOv5rt:
    def __init__(self, engine_path='yolov5m.engine', name='person', multiview=True,
                 conf_thres=0.3, iou_thres=0.45, input_size=640, device='cuda:0'):
        import torch, tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        self.device = device
        self.engine_path = engine_path
        self.name = name
        self.multiview = multiview
        self.output = 'output'
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size
        self.class_id = 0  # person in COCO
        self.torch = torch
        self.cuda = cuda

        # load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # allocate GPU buffers
        self.max_batch = 8  
        self.output_shape = (self.max_batch, 25200, 85)  # YOLOv5m 640x640
        self.d_input = cuda.mem_alloc(1 * 3 * self.input_size * self.input_size * 4 * self.max_batch)
        self.d_output = cuda.mem_alloc(np.empty(self.output_shape, dtype=np.float32).nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    # ---------------- Cache ----------------
    def dump(self, cachename, output):
        import os, pickle
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        with open(cachename, 'wb') as f:
            pickle.dump(output, f)
        return output

    def load(self, cachename):
        import pickle
        with open(cachename, 'rb') as f:
            return pickle.load(f)

    def check_cache(self, imgname):
        import os
        basename = os.path.basename(imgname)
        imgext = '.' + basename.split('.')[-1] if '.' in basename else ''
        nv = os.path.basename(os.path.dirname(imgname)) or 'root'
        cachename = os.path.join(self.output, self.name, nv, basename.replace(imgext, '.npy'))
        if os.path.exists(cachename):
            return True, self.load(cachename), cachename
        return False, None, cachename

    def check_image(self, img_or_name):
        import cv2
        if isinstance(img_or_name, str):
            img = cv2.imread(img_or_name)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_or_name}")
        else:
            img = img_or_name
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # ---------------- Batch TensorRT inference ----------------
    def detect_batch(self, images):
        import numpy as np
        images_proc = []
        orig_shapes = []
        for img in images:
            orig_shapes.append(img.shape)
            img_resized = cv2.resize(img, (self.input_size, self.input_size))
            img_input = img_resized.transpose(2,0,1).astype(np.float32)/255.0
            images_proc.append(img_input)
        input_batch = np.stack(images_proc, axis=0)
        batch_size = len(images)
        input_batch = np.ascontiguousarray(input_batch)
        output = np.empty((batch_size, 25200, 85), dtype=np.float32)

        self.cuda.memcpy_htod_async(self.d_input, input_batch, self.stream)
        self.context.execute_async_v2(self.bindings, stream_handle=self.stream.handle)
        self.cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()

        return output, orig_shapes

    # ---------------- GPU NMS ----------------
    def postprocess(self, output, orig_shapes):
        import torch
        boxes_all = []
        for i, pred in enumerate(output):
            pred = torch.tensor(pred).to(self.device)
            if pred.numel() == 0:
                boxes_all.append(torch.zeros((0,6), device=self.device))
                continue
            # conf * class score
            scores, labels = torch.max(pred[:,5:], dim=1)
            conf = pred[:,4] * scores
            mask = (conf > self.conf_thres) & (labels == self.class_id)

            pred = pred[mask]          # shape: [N, 85]
            conf = conf[mask]           # shape: [N]
            labels = labels[mask]       # shape: [N]

            if pred.shape[0] == 0:
                boxes_all.append(torch.zeros((0,6), device=self.device))
                continue

            x, y, w, h = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
            x1 = (x-w/2) * orig_shapes[i][1] / self.input_size
            y1 = (y-h/2) * orig_shapes[i][0] / self.input_size
            x2 = (x+w/2) * orig_shapes[i][1] / self.input_size
            y2 = (y+h/2) * orig_shapes[i][0] / self.input_size
            out_boxes = torch.stack([x1, y1, x2, y2, conf, labels], dim=1)

            keep = torch.ops.torchvision.nms(out_boxes[:,:4], out_boxes[:,4], self.iou_thres)
            boxes_all.append(out_boxes[keep])
        return boxes_all

    # ---------------- Callable ----------------
    def detect(self, image, imgname):
        flag, cache, cachename = self.check_cache(imgname)
        if flag:
            return cache
        image = self.check_image(image)
        output, shapes = self.detect_batch([image])
        boxes = self.postprocess(output, shapes)
        res = {'results': boxes[0].cpu().numpy(), 'image_shape': shapes[0]}
        self.dump(cachename, res)
        return res

    def __call__(self, images, imgnames, top_k=None):
        squeeze=False
        if not isinstance(images,list):
            images=[images]
            imgnames=[imgnames]
            squeeze=True

        detects={'bbox':[[] for _ in range(len(images))]}
        batch_imgs, cache_flags, cache_results, cache_names = [],[],[],[]

        # Step1: check cache
        for img,name in zip(images,imgnames):
            flag, cache, cachename = self.check_cache(name)
            cache_flags.append(flag)
            cache_results.append(cache)
            cache_names.append(cachename)
            batch_imgs.append(self.check_image(img) if not flag else None)

        # Step2: batch inference for all uncached images
        need_idx = [i for i,f in enumerate(cache_flags) if not f]
        if len(need_idx)>0:
            input_batch = [batch_imgs[i] for i in need_idx]
            output, shapes = self.detect_batch(input_batch)
            boxes_list = self.postprocess(output, shapes)
            for idx, i in enumerate(need_idx):
                boxes = boxes_list[idx].cpu().numpy()
                res = {'results': boxes, 'image_shape': shapes[idx]}
                self.dump(cache_names[i], res)
                cache_results[i] = res

        # Step3: postprocess + top_k
        for i, res in enumerate(cache_results):
            select,_ = res['results'], res
            if len(select)==0:
                select=np.zeros((0,5),dtype=np.float32)
            else:
                select=np.array(select).astype(np.float32)
                if top_k is not None:
                    idx = np.argsort(select[:,4])[::-1]
                    select = select[idx[:top_k]]
            detects['bbox'][i]=select

        if squeeze:
            detects['bbox']=detects['bbox'][0]
        return detects






class YoloWithTrack(BaseYOLOv5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.track_cache = {}

    @staticmethod
    def calculate_iou(bbox_pre, bbox_now):
        area_now = (bbox_now[:, 2] - bbox_now[:, 0])*(bbox_now[:, 3]-bbox_now[:, 1])
        area_pre = (bbox_pre[:, 2] - bbox_pre[:, 0])*(bbox_pre[:, 3]-bbox_pre[:, 1])
        # compute IOU
        # max of left
        xx1 = np.maximum(bbox_now[:, 0], bbox_pre[:, 0])
        yy1 = np.maximum(bbox_now[:, 1], bbox_pre[:, 1])
        # min of right
        xx2 = np.minimum(bbox_now[:, 0+2], bbox_pre[:, 0+2])
        yy2 = np.minimum(bbox_now[:, 1+2], bbox_pre[:, 1+2])
        # w h
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        over = (w*h)/(area_pre+area_now-w*h)
        return over

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        sub = os.path.basename(os.path.dirname(imgname))
        frame = int(os.path.basename(imgname).split('.')[0])
        if sub not in self.track_cache:
            # select the best
            select = super().select_bbox(select, results, imgname)
            self.track_cache[sub] = {
                'frame': [frame],
                'bbox': [select]
            }
            return select
        bbox_pre = self.track_cache[sub]['bbox'][-1]
        iou = self.calculate_iou(bbox_pre, select)
        idx = iou.argmax()
        select = select[idx:idx+1]
        self.track_cache[sub]['frame'].append(frame)
        self.track_cache[sub]['bbox'].append(select)
        return select

class MultiPerson(BaseYOLOv5):
    def __init__(self, min_length, max_length, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        print('[{}] Only keep the bbox in [{}, {}]'.format(self.__class__.__name__, min_length, max_length))

    # def select_bbox(self, select, results, imgname):
    #     # print("Before filtering:", select)
    #     if select.shape[0] == 0:
    #         return select
    #     # 判断一下面积
    #     area = np.sqrt((select[:, 2] - select[:, 0])*(select[:, 3]-select[:, 1]))
    #     valid = (area > self.min_length) & (area < self.max_length)
    #     height, width, _ = results['image_shape']
    #     valid = valid & (select[:, 2] > self.min_length * 1.5) & (select[:, 0] < width - self.min_length * 1.5)
    #     # print("After filtering:", select[valid])
    #     return select[valid]
    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select

        w = select[:,2] - select[:,0]
        h = select[:,3] - select[:,1]

        valid = (w > self.min_length) & (w < self.max_length) & \
                (h > self.min_length) & (h < self.max_length)
        
        # print("After filtering:", select[valid])

        return select[valid]
    
class YoloWithTrack_v11(BaseYOLOv11):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.track_cache = {}

    @staticmethod
    def calculate_iou(bbox_pre, bbox_now):
        # 確保都是 numpy array
        bbox_pre = np.array(bbox_pre, dtype=np.float32)
        bbox_now = np.array(bbox_now, dtype=np.float32)

        area_now = (bbox_now[:, 2] - bbox_now[:, 0]) * (bbox_now[:, 3] - bbox_now[:, 1])
        area_pre = (bbox_pre[:, 2] - bbox_pre[:, 0]) * (bbox_pre[:, 3] - bbox_pre[:, 1])

        xx1 = np.maximum(bbox_now[:, 0][:, None], bbox_pre[:, 0][None, :])
        yy1 = np.maximum(bbox_now[:, 1][:, None], bbox_pre[:, 1][None, :])
        xx2 = np.minimum(bbox_now[:, 2][:, None], bbox_pre[:, 2][None, :])
        yy2 = np.minimum(bbox_now[:, 3][:, None], bbox_pre[:, 3][None, :])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        iou = (w * h) / (area_now[:, None] + area_pre[None, :] - w * h + 1e-6)
        return iou  # shape: (N_now, N_pre)

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select.astype(np.float32)

        sub = os.path.basename(os.path.dirname(imgname))
        frame = int(os.path.basename(imgname).split('.')[0])

        select = np.array(select, dtype=np.float32)

        if sub not in self.track_cache:
            # 選擇最佳 bbox
            idx = np.argmax(select[:, 4])
            select_best = select[idx:idx + 1]
            self.track_cache[sub] = {
                'frame': [frame],
                'bbox': [select_best]
            }
            return select_best

        bbox_pre = np.array(self.track_cache[sub]['bbox'][-1], dtype=np.float32)
        iou = self.calculate_iou(bbox_pre, select)
        idx = iou.argmax()
        select_best = select[idx:idx + 1]

        self.track_cache[sub]['frame'].append(frame)
        self.track_cache[sub]['bbox'].append(select_best)

        return select_best


class MultiPerson_v11(BaseYOLOv11):
    def __init__(self, min_length, max_length, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        print(f'[{self.__class__.__name__}] Only keep the bbox in [{min_length}, {max_length}]')

    # def select_bbox(self, select, results, imgname):
    #     # print("Before filtering:", select)
    #     # return select
    #     if select.shape[0] == 0:
    #         return select
    #     # 判断一下面积
    #     area = np.sqrt((select[:, 2] - select[:, 0])*(select[:, 3]-select[:, 1]))
    #     valid = (area > self.min_length) & (area < self.max_length)
    #     height, width, _ = results['image_shape']
    #     valid = valid & (select[:, 2] > self.min_length * 1.5) & (select[:, 0] < width - self.min_length * 1.5)
    #     # print("After filtering:", select[valid])
    #     return select[valid]
    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select

        w = select[:,2] - select[:,0]
        h = select[:,3] - select[:,1]

        valid = (w > self.min_length) & (w < self.max_length) & \
                (h > self.min_length) & (h < self.max_length)

        # print("After filtering:", select[valid])
        return select[valid]






class DetectToPelvis:
    def __init__(self, key) -> None:
        self.key = key
        self.multiview = True
    
    def __call__(self, **kwargs):
        key = self.key
        val = kwargs[key]
        ret = {'pelvis': []}
        for nv in range(len(val)):
            bbox = val[nv]
            center = np.stack([(bbox[:, 0] + bbox[:, 2])/2, (bbox[:, 1] + bbox[:, 3])/2, bbox[:, -1]], axis=-1)
            ret['pelvis'].append(center)
        return ret

class Yolo_model:
    def __init__(self, mode, yolo_ckpt, multiview, repo_or_dir = 'ultralytics/yolov5', source='github') -> None:
        yolo_ckpt = check_modelpath(yolo_ckpt)
        self.model = torch.hub.load(repo_or_dir, 'custom', yolo_ckpt, source=source)
        self.min_detect_thres = 0.3
        self.mode = mode # 'fullimg' # 'bboxcrop'
        self.output = 'output'
        self.name = 'yolo'
        self.multiview = multiview
    @torch.no_grad()
    def det_step(self, img_or_name, imgname, bbox=[]):

        basename = os.path.basename(imgname)
        if self.multiview:
            nv = imgname.split('/')[-2]
            cachename = join(self.output, self.name, nv, basename.replace('.jpg', '.pkl'))
        else:
            cachename = join(self.output, self.name, basename.replace('.jpg', '.pkl'))
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        if os.path.exists(cachename):
            with open(cachename, 'rb') as f:
                output = pickle.load(f)
            return output

        if isinstance(img_or_name,str):
            images = cv2.imread(img_or_name)
        else:
            images = img_or_name

        if self.mode == 'bboxcrop':
            bbox[0] = max(0,bbox[0])
            bbox[1] = max(0,bbox[1])
            crop = images[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),::-1]
        else:
            crop = images[:,:,::-1]
        # print("[yolo img shape] ",crop.shape)
        results = self.model(crop) #RGB images[:,:,::-1]
        # breakpoint()
        arrays = np.array(results.pandas().xyxy[0])
        bboxes = {
            'bbox':[],
            'bbox_handl':[],
            'bbox_handr':[],
            'pelvis':[],
            'pelvis_l':[],
            'pelvis_r':[]
        }

        for i, res in enumerate(arrays):
            classid = res[5]
            box = res[:5]
            if self.mode == 'bboxcrop':
                box[0]+=bbox[0]
                box[2]+=bbox[0]
                box[1]+=bbox[1]
                box[3]+=bbox[1]
            if False:
                vis = images.copy()
                cpimg = crop.copy()
                from easymocap.mytools.vis_base import plot_bbox
                plot_bbox(vis,box,0)
                plot_bbox(cpimg,res[:5],0)
                cv2.imshow('vis',vis)
                # cv2.waitKey(0)
                cv2.imshow('crop',cpimg)
                cv2.waitKey(0)
                breakpoint()
            if box[4] < self.min_detect_thres:
                continue
            if classid==0:
                bboxes['bbox'].append(box)
            elif classid==1:
                bboxes['bbox_handl'].append(box)
                bboxes['pelvis_l'].append([(box[0]+box[2])/2,(box[1]+box[3])/2,box[-1]])
            elif classid==2:
                bboxes['bbox_handr'].append(box)
                bboxes['pelvis_r'].append([(box[0]+box[2])/2,(box[1]+box[3])/2,box[-1]])
        if(len(bboxes['bbox_handl'])==0):
            # bboxes['bbox_handl'].append(np.zeros((0, 5)))
            # bboxes['pelvis_l'].append(np.zeros((0, 3)))
            bboxes['bbox_handl'].append(np.zeros((5)))
            bboxes['pelvis_l'].append(np.zeros((3)))
            
        if(len(bboxes['bbox_handr'])==0):
            # bboxes['bbox_handr'].append(np.zeros((0, 5)))
            # bboxes['pelvis_r'].append(np.zeros((0, 3)))
            bboxes['bbox_handr'].append(np.zeros((5)))
            bboxes['pelvis_r'].append(np.zeros((3)))
        if(len(bboxes['bbox'])==0):
            bboxes['bbox'].append(np.zeros((5)))
        bboxes['bbox'] = np.array(bboxes['bbox'])
        if isinstance(imgname,str):
            with open(cachename, 'wb') as f:
                pickle.dump(bboxes, f)
        return bboxes
    def __call__(self, images, imgname, bbox=[]):
        return self.det_step(images, imgname, bbox)
    
class Yolo_model_v11:
    def __init__(self, mode, yolo_ckpt, multiview=True, device='cuda') -> None:
        """
        mode:       'bboxcrop' 或 'full'
        yolo_ckpt:  模型權重路徑 (ex: 'yolo11s-pose.pt')
        multiview:  是否啟用多視角 cache
        device:     推論裝置
        """
        yolo_ckpt = check_modelpath(yolo_ckpt)
        print(f"[Yolo_model_v11] Load model from: {yolo_ckpt}")
        self.model = YOLO(yolo_ckpt).to(device)

        self.device = device
        self.min_detect_thres = 0.3
        self.mode = mode
        self.output = 'output'
        self.name = 'yolo'
        self.multiview = multiview

    # --------------------------
    # Detection Step
    # --------------------------
    @torch.no_grad()
    def det_step(self, img_or_name, imgname, bbox=[]):
        # --- Cache 路徑 ---
        basename = os.path.basename(imgname)
        if self.multiview:
            nv = imgname.split('/')[-2]            # 假設上一層目錄為視角名
            cachename = join(self.output, self.name, nv, basename.replace('.jpg', '.pkl'))
        else:
            cachename = join(self.output, self.name, basename.replace('.jpg', '.pkl'))

        os.makedirs(os.path.dirname(cachename), exist_ok=True)

        # --- 讀取 Cache ---
        if os.path.exists(cachename):
            with open(cachename, 'rb') as f:
                print(f"[det_step] Load cache: {cachename}")
                return pickle.load(f)

        # --- 讀取圖片 ---
        if isinstance(img_or_name, str):
            image = cv2.imread(img_or_name)
        else:
            image = img_or_name

        # --- bboxcrop 模式 ---
        if self.mode == 'bboxcrop':
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            crop = image[int(bbox[1]):int(bbox[3]),
                         int(bbox[0]):int(bbox[2]),
                         ::-1]                      # BGR→RGB
        else:
            crop = image[:, :, ::-1]

        # --- YOLOv11 推論 ---
        results = self.model.predict(
            source=crop,
            conf=self.min_detect_thres,
            device=self.device,
            verbose=False
        )

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classids = r.boxes.cls.cpu().numpy()

        # --- 儲存檢測結果 ---
        bboxes = {
            'bbox': [],
            'bbox_handl': [],
            'bbox_handr': [],
            'pelvis': [],
            'pelvis_l': [],
            'pelvis_r': []
        }

        for i in range(len(boxes)):
            box = np.append(boxes[i], scores[i])
            cls = int(classids[i])

            # 如果是 bboxcrop 模式需要把座標還原到原圖
            if self.mode == 'bboxcrop':
                box[0] += bbox[0]; box[2] += bbox[0]
                box[1] += bbox[1]; box[3] += bbox[1]

            if cls == 0:
                bboxes['bbox'].append(box)
            elif cls == 1:
                bboxes['bbox_handl'].append(box)
                bboxes['pelvis_l'].append([(box[0]+box[2])/2, (box[1]+box[3])/2, box[4]])
            elif cls == 2:
                bboxes['bbox_handr'].append(box)
                bboxes['pelvis_r'].append([(box[0]+box[2])/2, (box[1]+box[3])/2, box[4]])

        # --- 保證輸出形狀 ---
        if len(bboxes['bbox_handl']) == 0:
            bboxes['bbox_handl'].append(np.zeros((5)))
            bboxes['pelvis_l'].append(np.zeros((3)))
        if len(bboxes['bbox_handr']) == 0:
            bboxes['bbox_handr'].append(np.zeros((5)))
            bboxes['pelvis_r'].append(np.zeros((3)))
        if len(bboxes['bbox']) == 0:
            bboxes['bbox'].append(np.zeros((5)))

        bboxes['bbox'] = np.array(bboxes['bbox'])

        # --- 存 Cache ---
        with open(cachename, 'wb') as f:
            pickle.dump(bboxes, f)

        print(f"[det_step] {imgname} detected {len(boxes)} objects")
        return bboxes

    # --------------------------
    # Call
    # --------------------------
    def __call__(self, images, imgname, bbox=[]):
        return self.det_step(images, imgname, bbox)


# class Yolo_model_hand_mvmp(Yolo_model):
#     @torch.no_grad()
#     def __call__(self, bbox, images, imgnames):
#         ret = {
#             'pelvis_l':[],
#             'pelvis_r':[],
#             # 'pelvis':[],
#             'bbox_handl':[],
#             'bbox_handr':[],
#         }
#         for nv in range(len(images)):
#             img = images[nv]
#             imgname = imgnames[nv]
#             if self.mode == 'bboxcrop':
#                 bboxes = {
#                     'bbox':[],
#                     'bbox_handl':[],
#                     'bbox_handr':[],
#                     'pelvis_l':[],
#                     'pelvis_r':[]
#                 }
#                 for pid in range(len(bbox[nv])):
#                     bboxes_ = self.det_step(img, imgname, bbox[nv][pid])
#                     for key in bboxes.keys():
#                         bboxes[key].append(bboxes_[key])
#             else:
#                 bboxes = self.det_step(img, imgname)
#             for k in ret.keys():
#                 ret[k].append(np.array(bboxes[k]))

#         return ret
class Yolo_model_hand_mvmp(Yolo_model_v11):
    @torch.no_grad()
    def __call__(self, bbox, images, imgnames):
        ret = {
            'pelvis_l':[],
            'pelvis_r':[],
            'bbox_handl':[],
            'bbox_handr':[],
        }
        for nv in range(len(images)):
            img = images[nv]
            imgname = imgnames[nv]
            if self.mode == 'bboxcrop':
                bboxes = {'bbox':[], 'bbox_handl':[], 'bbox_handr':[], 'pelvis_l':[], 'pelvis_r':[]}
                for pid in range(len(bbox[nv])):
                    bboxes_ = self.det_step(img, imgname, bbox[nv][pid])
                    for key in bboxes.keys():
                        # 確保 append 正確
                        if isinstance(bboxes_[key], list):
                            bboxes[key].extend(bboxes_[key])
                        else:
                            bboxes[key].append(bboxes_[key])
            else:
                bboxes = self.det_step(img, imgname)
                # 確保每個 key 都是 list
                for key in bboxes.keys():
                    if not isinstance(bboxes[key], list):
                        bboxes[key] = [bboxes[key]]
            # 將結果收集到 ret
            for k in ret.keys():
                ret[k].append(np.array(bboxes[k]))
        return ret
