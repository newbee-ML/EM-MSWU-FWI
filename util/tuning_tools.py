import itertools


def get_tuning_cmd(base_cmd, tuning_params_dict, fixed_params_dict, run_name, log_root):
    tuning_names = list(tuning_params_dict.keys())
    comb_list = itertools.product(*tuning_params_dict.values())
    fix_part = ' '.join(['--{} {}'.format(k, v) for k, v in fixed_params_dict.items()])
    cmd_list = []
    for comb in comb_list:
        tuning_comb = ' '.join(['--{} {}'.format(tuning_names[k], v) for k, v in enumerate(list(comb))])
        log_file = '_'.join(['{}={}'.format(tuning_names[k], v) for k, v in enumerate(list(comb))])
        cmd_list.append('%s %s %s > %s/%s-%s.log 2>&1 ' % (base_cmd, fix_part, tuning_comb, log_root, run_name, log_file))
    return cmd_list

def divide_group(list, group_num):
    elm_num = len(list) // group_num
    groups_list = []
    for i in range(0, len(list), elm_num):
        try:
            groups_list.append(list[i:i + elm_num])
        except IndexError:
            groups_list.append(list[i:])
    return groups_list

if __name__ == "__main__":
    para_dict = {
        'lr': [1e-2, 1e-3, 1e-4],
        'bs': [2, 3, 4],
        'loss': ['L2', 'L1', 'W']
    }
    fix_dict = {
        'device': 0,
        'save_group': 1
    }
    cmd_list = get_tuning_cmd('python train.py', para_dict, fix_dict, 'test', 'log')
    print(cmd_list)
    for cmds in divide_group(cmd_list, 4):
        print(len(cmds))