"""Provisioned MuJoCo check of direct stepping, 3+1 records and sensor synchronization."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from nablamimic.mujoco_source_adapter import MuJoCoSourceAdapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--iterations',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_binary_path(str(args.model))
    contract=json.loads(args.contract.read_text())
    reference=np.load(args.reference)
    q=torch.tensor(reference['qpos'][args.frame:args.frame+1],dtype=torch.float64)
    v=torch.tensor(reference['qvel'][args.frame:args.frame+1],dtype=torch.float64)
    warm=torch.zeros_like(v)
    control=torch.tensor(contract['source_action']['default_joint_position'],dtype=torch.float64)[None]
    friction=torch.ones(1,dtype=torch.float64)
    com=torch.zeros(1,3,dtype=torch.float64)
    adapter=MuJoCoSourceAdapter(model,iterations=args.iterations)
    def advance(owner,qpos,qvel,initial_warm,substeps):
        return owner.forward_measurements(qpos,qvel,initial_warm,control,friction,com,
            iterations=args.iterations,substeps=substeps,tape=True)
    four=advance(adapter,q,v,warm,4)
    direct=mujoco.MjData(model)
    direct.qpos[:]=q[0].numpy()
    direct.qvel[:]=v[0].numpy()
    direct.ctrl[:]=control[0].numpy()
    direct.qacc_warmstart[:]=warm[0].numpy()
    for _ in range(4):
        mujoco.mj_step(model,direct)
    split_owner=MuJoCoSourceAdapter(model,iterations=args.iterations)
    first=advance(split_owner,q,v,warm,3)
    last=advance(split_owner,first['qpos'],first['qvel'],first['warm'],1)
    sensor_owner=MuJoCoSourceAdapter(model,iterations=args.iterations)
    first_sensor=advance(sensor_owner,q,v,warm,3)
    sensor_owner.synchronized_data(first_sensor['qpos'],first_sensor['qvel'],first_sensor['warm'])
    sensor_last=advance(sensor_owner,first_sensor['qpos'],first_sensor['qvel'],first_sensor['warm'],1)
    comparisons={}
    for name,direct_name in (('qpos','qpos'),('qvel','qvel'),('warm','qacc_warmstart')):
        comparisons['direct_four_'+name]=np.array_equal(four[name][0].numpy(),getattr(direct,direct_name))
        comparisons['split_four_'+name]=torch.equal(four[name],last[name])
        comparisons['sensor_sync_'+name]=torch.equal(last[name],sensor_last[name])
    for name in ('terrain_wrench','terrain_contact_count','all_contact_wrench','all_contact_count',
                 'actuator_torque','substep_info','input_qpos','input_qvel'):
        comparisons['split_four_'+name]=torch.equal(four[name],torch.cat((first[name],last[name]),1))
    comparisons['double_state']=all(four[name].dtype==torch.float64 for name in ('qpos','qvel','warm'))
    comparisons['physical_clock']=all(np.isclose(owner.data_batch[0].time,.02)
        for owner in (adapter,split_owner,sensor_owner))
    adapter.reset_clock()
    comparisons['reset_clock']=adapter.substep_index==0 and adapter.data_batch[0].time==0.
    result=dict(comparisons=comparisons,passed=all(comparisons.values()),physics=adapter.metadata(),
                input_frame=args.frame,model=str(args.model),reference=str(args.reference))
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit('MuJoCo adapter consistency failed')


if __name__=='__main__':
    main()
