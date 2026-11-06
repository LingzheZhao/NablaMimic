"""Closed-loop learned-state actor refinement with an explicit finite-horizon loss.

This diagnostic uses source reference pose/velocity, termination margins and
executed-action smoothness. It has no critic or predicted contact-force reward.
The simulator independently measures the exported actor's actual execution.
"""
import torch

from .source_world_model_render import state_to_qpos_qvel


def policy_observation(history, depth):
    terms = [history[..., start:end].flatten(1) for start,end in
             ((0,3),(3,6),(6,35),(35,64),(64,93))]
    return torch.cat((*terms,depth.flatten(1)), -1)


class SourceWorldModelImagination:
    def __init__(self, env, context, model, vision, condition, objective, *, teacher=None):
        if condition not in ('render','latent') or model.latent_dim != (32 if condition=='latent' else 0):
            raise ValueError('Visual condition and world-model latent dimensions differ')
        self.env,self.context,self.model,self.vision = env,context,model,vision
        self.condition,self.objective,self.teacher = condition,objective,teacher
        self.teacher_action_error_units = 'PD setpoint radians squared after action_scale; normalized error reported separately'
        self.teacher_requires_additional_render = condition=='latent' and objective['teacher_action_weight']>0
        model.eval()
        vision.encoder.eval()
        vision.depth_decoder.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (vision.encoder,vision.depth_decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for parameter in vision.actor_layers.parameters():
            parameter.requires_grad_(True)
        if teacher is not None:
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        if objective['teacher_action_weight'] > 0 and teacher is None:
            raise ValueError('Teacher regularization requires an explicit frozen teacher')

    def physical_state(self, state, frame, previous_action, com):
        qpos,qvel = state_to_qpos_qvel(state)
        qnorm,position,quaternion,velocity,subtree_com = self.env.engine_kinematics(
            qpos.contiguous(),qvel.contiguous(),com.contiguous())
        return dict(qpos=qnorm,qvel=qvel,xpos=position,xquat=quaternion,cvel=velocity,
                    subtree_com=subtree_com,frame=frame,prev_a=previous_action,
                    enc_bias=qpos.new_zeros((qpos.shape[0],29)),com=com)

    def rollout(self, initial_state, history, previous_action, initial_frame, com, horizon):
        """Read only initial real state/history, then recursively predict feedback."""
        state = initial_state
        frame = initial_frame
        physical = self.physical_state(state,frame,previous_action,com)
        depth = self.context.render(state)
        observation = policy_observation(history,depth)
        goal = self.env.source_goal(physical)
        latent = (self.vision.encode(observation,goal)[0].detach() if self.condition=='latent'
                  else state[0].new_empty((state[0].shape[0],0)))
        losses,actions = [],[]
        teacher_normalized_total = state[0].new_zeros(())
        totals = {name:state[0].new_zeros(()) for name in
                  ('body_position','body_velocity','torso_orientation','joint_position','joint_velocity','termination','action_smoothness','teacher_action')}
        for step in range(horizon):
            if step:
                goal = self.env.source_goal(physical)
            if self.condition=='render':
                depth = self.context.render(state) if step else depth
                feature = self.vision.encode(policy_observation(history,depth),goal)[0].detach()
            else:
                feature = latent
            # Normalize the current predicted history and current-frame goal.
            _,normalized_goal,normalized_history = self.vision.encode(policy_observation(history,depth),goal)
            action = self.env.bound_action(self.vision.actor_from_features(feature,normalized_goal,normalized_history))
            teacher_loss = action.new_zeros(())
            teacher_normalized_loss = action.new_zeros(())
            if self.teacher is not None and self.objective['teacher_action_weight']>0:
                # Teacher uses its own original encoder and identical current sensor input.
                teacher_depth = self.context.render(state) if self.condition=='latent' else depth
                with torch.no_grad():
                    teacher_action = self.env.bound_action(self.teacher(policy_observation(history,teacher_depth),goal))
                teacher_normalized_loss = (action-teacher_action).square().mean()
                teacher_loss = ((action-teacher_action)*self.env.action_scale).square().mean()
            control = self.env.ctrl_of(action,frame)
            scan,physics = self.context(state)
            state,_,latent = self.model(state,control,scan,physics,latent)
            frame = frame+1
            following = self.physical_state(state,frame,action,com)
            actual_position = following['xpos'][:,self.env.body_ids]
            actual_velocity,_ = self.env.body_velocity(following)
            actual_quaternion = following['xquat'][:,self.env.torso_id]
            reference_quaternion = self.env.ref_body_quat[frame,self.env.anchor_k]
            terms = dict(body_position=(actual_position-self.env.ref_body_pos[frame]).square().mean(),
                body_velocity=(actual_velocity-self.env.ref_body_lin[frame]).square().mean(),
                torso_orientation=(1-(actual_quaternion*reference_quaternion).sum(-1).square()).clamp_min(0).mean(),
                joint_position=(following['qpos'][:,7:]-self.env.ref_qpos[frame,7:]).square().mean(),
                joint_velocity=(following['qvel'][:,6:]-self.env.ref_qvel[frame,6:]).square().mean(),
                termination=self.env.termination_penalty(following,self.objective['height_margin_start_m'],
                                                         self.objective['gravity_margin_start']).mean(),
                action_smoothness=(action-previous_action).square().mean(),teacher_action=teacher_loss)
            losses.append(sum(terms[name]*self.objective[name+'_weight'] for name in terms))
            teacher_normalized_total = teacher_normalized_total+teacher_normalized_loss
            for name,value in terms.items():
                totals[name] = totals[name]+value
            # Append differentiable predicted proprioception and the executed action.
            history = torch.cat((history[:,1:],self.env.proprio(following)[:,None]),1)
            previous_action = action
            physical = following
            actions.append(action)
        loss = torch.stack(losses).mean()
        metrics = {name:value/horizon for name,value in totals.items()}
        metrics['teacher_action_normalized'] = teacher_normalized_total/horizon
        return loss,metrics,dict(state=state,history=history,actions=torch.stack(actions,1),frame=frame)
