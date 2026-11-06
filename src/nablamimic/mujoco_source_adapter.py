"""Evaluation-only MuJoCo physics and measurements for the source tracking task."""
import numpy as np
import torch


class MuJoCoSourceAdapter:
    def __init__(self, model, *, iterations, applied_body_wrench_world=None):
        import mujoco
        self.model=model
        if model.opt.integrator!=mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            raise ValueError('Source MuJoCo evaluation requires the prepared implicitfast integrator')
        if model.na:
            raise ValueError('Source MuJoCo evaluation requires stateless actuators')
        if not np.isclose(model.opt.timestep,.005):
            raise ValueError('Source MuJoCo evaluation requires 200 Hz physics')
        self.original_iterations=int(model.opt.iterations)
        model.opt.iterations=iterations
        self.data_batch=()
        self.applied_body_wrench_world=applied_body_wrench_world
        self.substep_index=0
        self.mujoco=mujoco

    def reset_clock(self):
        self.substep_index=0
        for data in self.data_batch:
            data.time=0.

    def _allocate(self, batch):
        if len(self.data_batch)!=batch:
            self.data_batch=tuple(self.mujoco.MjData(self.model) for _ in range(batch))

    def synchronized_data(self, qpos, qvel=None, warm=None):
        self._allocate(len(qpos))
        for world,data in enumerate(self.data_batch):
            data.qpos[:]=qpos[world].detach().cpu().numpy()
            if qvel is not None:
                data.qvel[:]=qvel[world].detach().cpu().numpy()
            saved=(data.qacc_warmstart.copy() if warm is None else warm[world].detach().cpu().numpy().copy())
            self.mujoco.mj_forward(self.model,data)
            data.qacc_warmstart[:]=saved
        return self.data_batch

    def kinematics(self, qpos, qvel, com):
        if bool(com.any()):
            raise ValueError('MuJoCo source evaluation requires nominal COM parameters')
        rows=self.synchronized_data(qpos,qvel)
        def tensor(name):
            return torch.as_tensor(np.stack([getattr(data,name).copy() for data in rows]),device=qpos.device,dtype=torch.float64)
        return tuple(tensor(name) for name in ('qpos','xpos','xquat','cvel','subtree_com'))

    def _wrenches(self, data):
        model=self.model
        all_wrench=np.zeros((model.nbody,6))
        terrain_wrench=np.zeros_like(all_wrench)
        all_count=np.zeros(model.nbody,dtype=np.int32)
        terrain_count=np.zeros_like(all_count)
        local=np.zeros(6)
        for index in range(data.ncon):
            contact=data.contact[index]
            if contact.efc_address<0:
                continue
            self.mujoco.mj_contactForce(model,data,index,local)
            rotation=np.asarray(contact.frame).reshape(3,3)
            force=rotation.T@local[:3]
            torque=rotation.T@local[3:]
            moment=torque+np.cross(contact.pos,force)
            bodies=(int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2]))
            for body,sign in ((bodies[0],-1.),(bodies[1],1.)):
                all_wrench[body]+=sign*np.concatenate((force,moment))
                all_count[body]+=1
                if 0 in bodies:
                    terrain_wrench[body]+=sign*np.concatenate((force,moment))
                    terrain_count[body]+=1
        return terrain_wrench,terrain_count,all_wrench,all_count

    def forward_measurements(self,qpos,qvel,warm,control,friction,com,*,iterations,substeps,tape=False):
        if iterations!=self.model.opt.iterations or bool(com.any()) or not bool((friction==1).all()):
            raise ValueError('MuJoCo source evaluation requires its fixed solver and nominal domain parameters')
        self._allocate(len(qpos))
        records={name:[] for name in ('terrain_wrench','terrain_contact_count','all_contact_wrench','all_contact_count',
            'actuator_torque','substep_info','input_qpos','input_qvel','applied_body_wrench_world','body_com_world')}
        for world,data in enumerate(self.data_batch):
            data.qpos[:]=qpos[world].detach().cpu().numpy()
            data.qvel[:]=qvel[world].detach().cpu().numpy()
            data.qacc_warmstart[:]=warm[world].detach().cpu().numpy()
            data.ctrl[:]=control[world].detach().cpu().numpy()
            rows={name:[] for name in records}
            for substep in range(substeps):
                rows['input_qpos'].append(data.qpos.copy())
                rows['input_qvel'].append(data.qvel.copy())
                data.xfrc_applied[:]=0.
                if self.applied_body_wrench_world is not None:
                    self.applied_body_wrench_world(world,self.substep_index+substep,data)
                rows['applied_body_wrench_world'].append(data.xfrc_applied.copy())
                self.mujoco.mj_step(self.model,data)
                terrain,count,all_force,all_count=self._wrenches(data)
                for name,value in (('terrain_wrench',terrain),('terrain_contact_count',count),
                        ('all_contact_wrench',all_force),('all_contact_count',all_count),
                        ('actuator_torque',data.qfrc_actuator[self.model.jnt_dofadr[self.model.actuator_trnid[:,0]]]),
                        ('substep_info',np.array([int(data.solver_niter[0]),data.ncon,data.nefc,0],dtype=np.int32)),
                        ('body_com_world',data.xipos)):
                    rows[name].append(np.array(value,copy=True))
            for name in records:
                records[name].append(np.stack(rows[name]))
        self.substep_index+=substeps
        def tensor(values):
            values=np.asarray(values)
            dtype=torch.int32 if np.issubdtype(values.dtype,np.integer) else torch.float64
            return torch.as_tensor(values,device=qpos.device,dtype=dtype)
        output={name:tensor(value) for name,value in records.items()}
        output.update(qpos=tensor([data.qpos.copy() for data in self.data_batch]),
            qvel=tensor([data.qvel.copy() for data in self.data_batch]),
            warm=tensor([data.qacc_warmstart.copy() for data in self.data_batch]))
        if tape:
            output['tape']={name:output[name] for name in ('input_qpos','input_qvel')}
        return output

    @staticmethod
    def read_substep_inputs(tape,*,substeps):
        return {'qpos':tape['input_qpos'],'qvel':tape['input_qvel']}

    def step(self,qpos,qvel,warm,control,friction,com,*,iterations,substeps,kinematics,measurements,**kwargs):
        measured=self.forward_measurements(qpos,qvel,warm,control,friction,com,
            iterations=iterations,substeps=substeps)
        result=(measured['qpos'],measured['qvel'],measured['warm'])
        if kinematics:
            result=(*result,*self.kinematics(measured['qpos'],measured['qvel'],com))
        return (*result,measured) if measurements else result

    def metadata(self):
        opt=self.model.opt
        return dict(backend='mujoco',mujoco_version=self.mujoco.__version__,physics_hz=1/opt.timestep,
            control_substeps=4,integrator=int(opt.integrator),solver=int(opt.solver),iterations=int(opt.iterations),
            original_model_iterations=self.original_iterations,tolerance=float(opt.tolerance),
            cone=int(opt.cone),jacobian=int(opt.jacobian),warmstart='Explicit qacc_warmstart state carried across controls',
            measurement_time='Input t_k state and forces/torques computed by mj_step before integration to t_(k+1)',
            physical_state_dtype='float64; actor and native ray inputs are explicit float32 views',
            substep_info_columns=['solver_iterations','contacts','constraint_rows','native_unsupported_not_applicable_zero'],
            wrench='World force and moment about world origin, including world reaction; all contacts include self contact')


from .source_perceptive_env import SourcePerceptiveEnv


class MuJoCoSourceEnv(SourcePerceptiveEnv):
    """Source task equations with double physical state and MuJoCo stepping."""
    def reset(self, frame, noise=None):
        state=super().reset(frame,noise)
        state['qvel']=state['qvel'].to(torch.float64)
        state['warm']=state['warm'].to(torch.float64)
        state['hist']=state['hist'].to(torch.float32)
        return state
