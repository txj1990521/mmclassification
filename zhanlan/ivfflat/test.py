import numpy as np, os
img_paths = np.load(r"D:\zhanlan\faiss_database_hybrid_new_data\global_img_paths.npy", allow_pickle=True)
target = "4.5 TL05027_label0_score0.995.png"
hits = [i for i,p in enumerate(img_paths) if os.path.basename(str(p)) == target]
print("in_index?", len(hits)>0, "ids:", hits[:10])
