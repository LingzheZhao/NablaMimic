"""Generated scalar objectives check the gradient diagnostic's identity boundary."""
import importlib.util
from pathlib import Path
import unittest

import torch

path=Path(__file__).parents[1]/'scripts/diagnose_source_action_gradients.py'
spec=importlib.util.spec_from_file_location('gradient_diagnostic',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GradientDiagnosticTest(unittest.TestCase):
    def test_identity_preserves_values_and_probe_dependency(self):
        action=torch.tensor([[-0.,1.,-2.]],requires_grad=True)
        probe=torch.zeros(2,requires_grad=True)
        output=module.IdentityCotangent.apply(action,probe)
        self.assertTrue(torch.equal(output.view(torch.int32),action.view(torch.int32)))
        gradient,statistics=torch.autograd.grad(output.square().sum(),(action,probe))
        torch.testing.assert_close(gradient,2*action)
        torch.testing.assert_close(statistics,torch.zeros_like(probe))

    def test_detaching_terms_decomposes_same_forward_objective(self):
        gradients=[]
        values=[]
        for selected in (None,0,1,2):
            action=torch.tensor([.2,-.4],requires_grad=True)
            terms=(action.square().sum(),action.sin().sum(),action.prod())
            loss=sum(value if selected is None or index==selected else value.detach()
                     for index,value in enumerate(terms))
            values.append(loss.detach())
            gradients.append(torch.autograd.grad(loss,action)[0])
        self.assertTrue(all(torch.equal(values[0],value) for value in values))
        torch.testing.assert_close(gradients[0],sum(gradients[1:]))
        reading=module.gradient_reading(gradients[0],gradients[0],torch.tensor([.5,.3]))
        self.assertAlmostEqual(reading['cosine_with_full'],1.,places=6)
        self.assertAlmostEqual(reading['normalized_setpoint_direction_rms'],1.,places=6)


if __name__=='__main__':
    unittest.main()
