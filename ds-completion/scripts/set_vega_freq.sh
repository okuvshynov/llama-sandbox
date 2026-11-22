L=6  # or 3                                                             
for c in 0 1 2 3; do                                                     
  D=/sys/class/drm/card$c/device                                         
  echo manual | sudo tee $D/power_dpm_force_performance_level > /dev/null
  echo $L     | sudo tee $D/pp_dpm_sclk > /dev/null                      
  echo "card$c: $(grep '\*' $D/pp_dpm_sclk)"                             
done                                                                     
