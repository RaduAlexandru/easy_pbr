#!/usr/bin/env python3

try:
  import torch
except ImportError:
  pass
from easypbr  import *

config_file="./config/subsurface_scattering.cfg"

view=Viewer.create(config_file)
#hide the gird floor
Scene.set_floor_visible(False)


def make_figure():
  #download model from https://www.3dscanstore.com/blog/Free-3D-Head-Model
  head=Mesh("/media/rosu/Data/data/3d_objs/3d_scan_store/OBJ/Head/Head.OBJ")
  head.set_diffuse_tex("/media/rosu/Data/data/3d_objs/3d_scan_store/JPG Textures/Head/JPG/Colour_8k.jpg", 1)
  head.set_normals_tex("/media/rosu/Data/data/3d_objs/3d_scan_store/JPG Textures/Head/JPG/Normal Map_SubDivision_1.jpg", 1)
  # head.set_gloss_tex("/media/rosu/Data/data/3d_objs/3d_scan_store/JPG Textures/Head/JPG/Gloss_8k.jpg")
  head.m_vis.m_roughness=0.55
  head.model_matrix.rotate_axis_angle([0,1,0], -80)
  head.apply_model_matrix_to_cpu(True)

  # jacket=Mesh("/media/rosu/Data/data/3d_objs/3d_scan_store/OBJ/Jacket/Jacket.OBJ")
  # jacket.set_diffuse_tex("/media/rosu/Data/data/3d_objs/3d_scan_store/JPG Textures/Jacket/JPG/Jacket_Colour.jpg", 1)
  # jacket.set_normals_tex("/media/rosu/Data/data/3d_objs/3d_scan_store/JPG Textures/Jacket/JPG/Jacket_Normal.jpg", 1)
  # # jacket.set_gloss_tex("/media/rosu/Data/data/3d_objs/3d_scan_store/JPG Textures/Jacket/JPG/Jacket_Gloss.jpg")
  # jacket.m_vis.m_roughness=0.55
  # jacket.model_matrix.rotate_axis_angle([0,1,0], -80)
  # jacket.apply_model_matrix_to_cpu(True)

  # jacket.name="jacket"
  # head.add_child(jacket)

  return head



# view.load_environment_map("./data/sibl/Barcelona_Rooftops/Barce_Rooftop_C_3k.hdr")
# view.load_environment_map("./data/hdr/blaubeuren_night_4k.hdr")
view.load_environment_map("./data/sibl/Desert_Highway/Road_to_MonumentValley_Ref.hdr")
# view.load_environment_map("/media/rosu/Data/data/hdri_haven/nature/epping_forest_01_4k.hdr")
view.m_ambient_color_power=0.2

# view.m_camera.from_string("-0.644804  0.302574  0.371758 -0.0450536  -0.476531 -0.0244621 0.877661 -0.00559545    0.224117  -0.0433487 30 0.0320167 32.0167")
# view.m_camera.from_string(" -1.20724  0.289488 -0.515533 -0.0146836  -0.832987 -0.0221319 0.552652 -0.00760287     0.22025 -0.00940612 32 0.0320167 32.0167")
view.m_camera.from_string("-1.38942 0.290353 -0.29555 -0.0148244  -0.778977 -0.0184295 0.626601 -0.0112228    0.22355 0.00683358 32 0.0320167 32.0167")

for i in range(1):
  head = make_figure()
  head.model_matrix.set_translation([i*0.5, 0, 0])
  head.m_vis.m_needs_sss=True
  head.compute_embree_ao(100)
  view.m_get_ao_from_precomputation=True
  Scene.show(head,"head"+str(i))

view.m_camera.m_exposure=1.4
view.spotlight_with_idx(0).m_power=0
# view.spotlight_with_idx(0).m_power=6
# view.spotlight_with_idx(0).from_string(" 0.627251  0.860619 -0.720104 -0.101852   0.89552  0.267949 0.3404 -0.0170008   0.223146 0.00489488 40 0.116063 116.063")

view.spotlight_with_idx(1).m_power=5
view.spotlight_with_idx(1).m_color=[255/255, 225/255, 225/255]
view.spotlight_with_idx(2).m_power=0













while True:
    view.update()
