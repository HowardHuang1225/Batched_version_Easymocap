data=data/zju-ls-feng
# 0. extract the video to images
python scripts/preprocess/extract_video.py ${data} --handface
# 2.1 example for SMPL reconstruction
python apps/demo/mv1p.py ${data} --out ${data}/output/smpl --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --vis_smpl
# 2.2 example for SMPL-X reconstruction
python apps/demo/mv1p.py ${data} --out ${data}/output/smplx --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body bodyhandface --model smplx --gender male --vis_smpl
# 2.3 example for MANO reconstruction
#     MANO model is required for this part
python apps/demo/mv1p.py ${data} --out ${data}/output/manol --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body handl --model manol --gender male --vis_smpl
python apps/demo/mv1p.py ${data} --out ${data}/output/manor --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body handr --model manor --gender male --vis_smpl