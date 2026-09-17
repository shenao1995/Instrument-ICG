"""Canonical normals and same-raster GPU attribute regression checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import trimesh
from icg.geometry import InstrumentMesh, SemanticRenderer, PARTS, part_rotations
from icg.kinematics import exp_so3


class GeometryDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        root=Path(cls.tmp.name)
        for name in ('shaft','wrist','left_gripper','right_gripper'):
            trimesh.creation.box(extents=[.01,.004,.004]).export(root/(name+'.obj'))
        cls.mesh=InstrumentMesh(root,convention='simulated_dual_arm_v1',arms=2,load_appearance=False)
        cls.appearance=InstrumentMesh(root,convention='simulated_dual_arm_v1',arms=2,load_appearance=True,
                                      part_albedo={name:[.5,.5,.5] for name in PARTS})
        cls.pose=dict(R=torch.tensor(np.stack([exp_so3([.1,.2,.1]),exp_so3([-.1,-.2,.05])])[None],dtype=torch.float32),
                      t=torch.tensor([[[-.01,0.,.08],[.01,0.,.09]]]),
                      joints=torch.tensor([[[.2,.3,.4],[-.2,.5,.6]]]))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_geometry_normals_exist_without_appearance(self):
        torch.testing.assert_close(self.mesh.vertices,self.appearance.vertices)
        torch.testing.assert_close(self.mesh.normals,self.appearance.normals)
        torch.testing.assert_close(self.mesh.normals.norm(dim=1),torch.ones(len(self.mesh.normals)))
        self.assertFalse(hasattr(self.mesh,'face_albedo'))

    def test_articulated_normal_rotations(self):
        rotations=part_rotations(self.pose)
        canonical=self.mesh.normals.split(self.mesh.counts)
        expected=torch.cat([n@rotations[name][:,arm].transpose(-1,-2)
                            for arm in range(2) for name,n in zip(PARTS,canonical)],1)
        actual=self.mesh.camera_normals(self.pose)
        torch.testing.assert_close(actual,expected)
        shifted={**self.pose,'t':self.pose['t']+.03}
        torch.testing.assert_close(actual,self.mesh.camera_normals(shifted))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA renderer required')
    def test_same_raster_attributes_unit_normals_zero_background(self):
        mesh=self.mesh.to('cuda')
        try:
            renderer=SemanticRenderer(mesh,(64,64),'nvdiffrast',supersample=1,render_rgb=False)
            pose={key:value.to('cuda') for key,value in self.pose.items()}
            k=torch.tensor([[160.,0.,31.5],[0.,160.,31.5],[0.,0.,1.]],device='cuda')
            with patch.object(renderer.dr,'rasterize',wraps=renderer.dr.rasterize) as raster:
                result=renderer.render_icg(pose,k)
            self.assertEqual(raster.call_count,1)
            visible=result['mask'].amax(1)>.5
            self.assertTrue(visible.any())
            self.assertEqual(result['normal'].shape,result['xyz'].shape)
            torch.testing.assert_close(result['normal'].norm(dim=1)[visible],torch.ones_like(result['normal'].norm(dim=1)[visible]))
            self.assertTrue((result['normal'].permute(0,2,3,1)[~visible]==0).all())
            self.assertTrue((result['xyz'].permute(0,2,3,1)[~visible]==0).all())
        finally:
            self.mesh.cpu()


if __name__=='__main__':unittest.main()
