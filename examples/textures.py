#!/usr/bin/env python3

try:
  import torch
except ImportError:
  pass
from easypbr  import *

config_file="./config/textures.cfg"

view=Viewer.create(config_file) 

#lantern
mesh=Mesh("./data/textured/lantern/lantern_obj.obj")
mesh.set_diffuse_tex("./data/textured/lantern/textures/lantern_Base_Color.jpg")
mesh.set_metalness_tex("./data/textured/lantern/textures/lantern_Metallic.jpg")
mesh.set_roughness_tex("./data/textured/lantern/textures/lantern_Roughness.jpg")


mesh.m_vis.set_color_texture()
Scene.show(mesh,"mesh")

#hide the gird floor
Scene.set_floor_visible(False)

while True:
    view.update()