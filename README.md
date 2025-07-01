# :sunny::sunglasses: Summer practice tasks

## Magnetic field and its Curl Flow calculation  

**Description**  

1. Convert `.sav` files to `.np`.
2. In the vicinity of radius `R` of any given point `O: {Ox, Oy, Oz}` calculate the mean magnetic field `B_mean`.
3. Use this vector `B_mean` as a plane normal `B_n`.  
4. Init a new local cylindrical frame with the origin `O` and `B_n` as z-axis.  
5. Plot a 2D-distribution of `B_n`, `B_r`, `B_phi` in the local coordinate system.  
6. Plot a 2D-distribution of `curl_B_n`, `curl_B_r`, `curl_B_phi` in the same frame.  
7. Write a python script for paraview, which conviniently visualizes stream tracer of radius `R` at point `O` and performs calculation of pipeline above. Then displaying the slicing plane.  
8. Propose a numerical criteria to best choose `R` for a twisted magnetic flux `B` rope.  


### Appendix  

Use `convert.py` to convert `.sav` to `.vtk`. In file `src/sav2vtk.py` you can find a function to convert `.sav` to `.np`.  

Some illustrations given in `/assets` dir.  

`.sav` file could be downloaded via link [Yandex.Drive](https://disk.yandex.ru/d/UjtpEJs-s5aEpA)  
