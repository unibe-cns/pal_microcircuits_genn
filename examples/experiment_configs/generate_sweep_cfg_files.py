import os
import yaml

def load_config(path):
    with open(path) as f:
        data = yaml.load_all(f, Loader=yaml.Loader)
        all_configs = next(iter(data))
    meta_params = all_configs[0]
    network_params = all_configs[1]
    return meta_params, network_params

def save_config(dirname, filename, meta_params, network_params):
    if not dirname[-1] == '/':
        dirname += '/'
    try:
        os.makedirs(dirname)
        print("Directory ", dirname, " Created ")
    except FileExistsError:
        print("Directory ", dirname, " already exists")
    # save parameter configs
    with open(dirname + filename, 'w') as f:
        yaml.dump([meta_params, network_params], f)
    return


if __name__ == '__main__':
    import sys
    base_cfg = sys.argv[1]
    save_path = sys.argv[2]
    name = sys.argv[3]
    seeds = [123456 + i for i in range(20)]
    meta_base, network_base = load_config(base_cfg)
    meta = meta_base.copy()
    for seed in seeds:
        filename = name + '_seed_{0}.yaml'.format(seed)
        meta['numpy_seed'] = seed
        meta['name'] = name + '_seed_{0}'.format(seed)
        save_config(save_path, filename, meta, network_base)
