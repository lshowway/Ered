import os
import torch



data_dir = "../knowledge_resource/wikipedia"
write_dir = data_dir + '/DD'
read_dir = data_dir + '/CC'

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

# load
files = os.listdir(read_dir)
print(files)
for file in files:
    if os.path.exists(os.path.join(read_dir, file)):
        features = torch.load(os.path.join(read_dir, file))
        print(file, len(features))
        if len(features) < 500000:
            torch.save(features, os.path.join(write_dir, file))
        else:
            num = len(features) // 500000 + 1
            for i in range(num):
                start = i * 500000
                end = (i+1) * 500000
                tmp = features[start: end]
                torch.save(tmp, os.path.join(write_dir, file + '_%s'%i))
    # else:
    #     print(111)