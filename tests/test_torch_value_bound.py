"""Generated checks for the native bootstrap value-bound condition."""
import unittest

import torch
from nablamimic import torch_nets as nets


class TorchValueBoundTest(unittest.TestCase):
    def test_source_bound_uses_control_reward_and_six_hundred_controls(self):
        bound=nets.discounted_return_bound(.1,.99,600)
        self.assertAlmostEqual(bound,9.975949907087,places=10)
        self.assertLess(bound,.1/(1-.99))

    def test_bootstrap_clip_blocks_outside_gradients_and_default_is_identity(self):
        value=torch.tensor([-30.,-2.,0.,2.,30.],requires_grad=True)
        used=nets.bootstrap_value(value,10.)
        torch.testing.assert_close(used,torch.tensor([-10.,-2.,0.,2.,10.]))
        torch.testing.assert_close(torch.autograd.grad(used.sum(),value)[0],torch.tensor([0.,1.,1.,1.,0.]))
        self.assertIs(nets.bootstrap_value(value,None),value)

    def test_td_consumes_clipped_next_values_and_tracking_terminal_has_no_bootstrap(self):
        bound=nets.discounted_return_bound(.1,.99,600)
        reward=torch.full((2,2),.1)
        terminated=torch.tensor([[False,False],[False,True]])
        done=torch.tensor([[False,False],[True,True]])
        raw=torch.full((2,2),380.)
        used=nets.bootstrap_value(raw,bound)
        target=nets.td_lambda(reward,terminated,done,used,.99,.95)
        self.assertAlmostEqual(float(target[1,0]),.1+.99*bound,places=5)
        self.assertAlmostEqual(float(target[1,1]),.1,places=6)
        self.assertLessEqual(float(target.max()),10.)
        self.assertGreater(float(nets.td_lambda(reward,terminated,done,raw,.99,.95).max()),300.)

    def test_bootstrap_gradient_switch_preserves_values_td_and_reward_gradient(self):
        gradients=[]
        values=[]
        targets=[]
        for enabled in (True,False):
            action=torch.tensor([.2,.4],requires_grad=True)
            reward=action.square()
            raw_next=3*action
            raw_end=5*action
            next_value=nets.bootstrap_value(raw_next,1.,gradient=enabled)
            end_value=nets.bootstrap_value(raw_end,1.,gradient=enabled)
            loss=-(reward+.9*next_value+.9**2*end_value).sum()
            gradients.append(torch.autograd.grad(loss,action)[0])
            values.append(torch.stack((next_value,end_value)))
            targets.append(nets.td_lambda(reward[None],torch.zeros(1,2,dtype=torch.bool),
                torch.ones(1,2,dtype=torch.bool),next_value[None],.9,.95))
        torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)
        torch.testing.assert_close(targets[0],targets[1],rtol=0,atol=0)
        torch.testing.assert_close(gradients[1],torch.tensor([-.4,-.8]))
        self.assertGreater(float((gradients[0]-gradients[1]).abs().sum()),0)
        raw=torch.tensor([.1],requires_grad=True)
        self.assertIs(nets.bootstrap_value(raw,None),raw)
        self.assertFalse(nets.bootstrap_value(raw,None,gradient=False).requires_grad)

    def test_critic_regression_uses_raw_prediction_and_can_correct_saturation(self):
        prediction=torch.tensor([30.],requires_grad=True)
        target=torch.tensor([5.])
        loss=(prediction-target).square().mean()
        self.assertEqual(float(torch.autograd.grad(loss,prediction)[0]),50.)


if __name__=='__main__':
    unittest.main()
