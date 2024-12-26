import os
from pathlib import Path

def get_lsf_head(job_name, queue_name, cpu_num, gpu_num, log_root):
    cmds = '#!/bin/bash\n'
    cmds += '#BSUB -J {}\n'.format(job_name)
    cmds += '#BSUB -q {}\n'.format(queue_name)
    cmds += '#BSUB -n {}\n'.format(cpu_num)
    cmds += '#BSUB -o {}/{}.out\n'.format(log_root, job_name.strip('.lsf'))
    cmds += '#BSUB -e {}/{}.err\n'.format(log_root, job_name.strip('.lsf'))
    cmds += '#BSUB -R "span[ptile={}]" \n'.format(cpu_num)
    cmds += '#BSUB -gpu "num={}/host" \n'.format(gpu_num)
    return cmds

def concat_lsf_file(lsf_file, lsf_header, root_path, conda_env, excute_cmd_list):
    os.makedirs(os.path.dirname(lsf_file), exist_ok=True)
    path = Path(lsf_file)
    with path.open('w', newline='', encoding='utf-8') as f:
        f.write(lsf_header)
        f.write('cd {}\n'.format(root_path))
        f.write('conda init\n')
        f.write('source activate\n')
        f.write('conda activate {}\n'.format(conda_env))
        f.write('date\n')
        for cmd in excute_cmd_list:
            f.write('{}\n'.format(cmd))
        f.write('date\n')