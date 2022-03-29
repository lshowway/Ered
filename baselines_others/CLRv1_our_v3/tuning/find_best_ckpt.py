import json
import sys

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        return config


if __name__ == "__main__":
    out_file = sys.argv[1]
    best_ckpt = None
    for i in range(2, len(sys.argv)):
        ckpt_info = load_config(sys.argv[i])
        ckpt_info = ckpt_info["ckpt_info"] if "ckpt_info" in ckpt_info else ckpt_info
        if best_ckpt is None:
            best_ckpt = {}
            best_ckpt["ckpt_id"] = i - 1
            best_ckpt["ckpt_info"] = ckpt_info
        elif ckpt_info["roc_auc"] > best_ckpt["ckpt_info"]["roc_auc"]:
            best_ckpt["ckpt_id"] = i - 1
            best_ckpt["ckpt_info"] = ckpt_info
    with open(out_file, 'w') as f:
        json.dump(best_ckpt, f)