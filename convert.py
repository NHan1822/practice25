from src.sav2vtk import box2vtk

box2vtk("~/data/20170904_test/data/NORH_NLFFFE_170904_055842.sav", field_name="Bnlfffe")

# NLFFFE means non-linear force-free field extrapolation
# https://arxiv.org/pdf/1208.4693

# Calculations of force-free fields performed by different algorithm, library, see
# https://github.com/Sergey-Anfinogentov/GXBox_prep  

# .SAV files usually provided as input.
# They contain extrapolated Magnetic Field (see above) and basemaps sometimes

# .vtk and .vtr files used for visualization purposes  
# .np files could be used for analysis with python  


