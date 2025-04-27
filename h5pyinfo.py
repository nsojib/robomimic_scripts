import h5py
import numpy as np

def print_dataset_info(dataset_path, close_the_file=True): 
    print('opening dataset: ', dataset_path)
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())
    
    lengths=[]
    for demo_name in demos:
        demo=f['data'][demo_name]
        num_samples=demo.attrs['num_samples']
        lengths.append(num_samples)

    lengths=np.array(lengths)

    print('Number of demos: ', len(demos))
    print('Max length: ', np.max(lengths))
    print('Min length: ', np.min(lengths))
    print('Mean length: ', np.mean(lengths))
    
    print('------mask keys------')
    for key in f['mask'].keys():
        print(key, f['mask'][key].shape)
        
    if close_the_file:
        f.close()
        
    return f
    