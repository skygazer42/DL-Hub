from __future__ import annotations
from ._common import build_toy_mesh, smoke_test_mesh
_VARIANTS = {'spec_hmr_tiny': {'width':24,'depth':1}, 'spec_hmr_small': {'width':32,'depth':2}, 'spec_hmr_base': {'width':48,'depth':3}}

def build_spec_hmr_mesh_recoverer(*, in_channels:int, variant:str='spec_hmr_small', width_mult:float=1.0, num_vertices:int=32):
    return build_toy_mesh(family='spec_hmr', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), num_vertices=int(num_vertices))

if __name__ == '__main__': smoke_test_mesh(build_spec_hmr_mesh_recoverer, 'spec_hmr_tiny')
