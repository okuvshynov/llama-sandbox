# The GPU clock of the four Vega II dies fixed at one level (needs sudo): L below, 6 = 1608 MHz,
# the level of every measurement since step 90. After a boot, before any GPU program (README steps
# 178-179; QUICKSTART "What it needs"); a reboot puts the cards back on auto.
L=6  # or 3                                                             
for c in 0 1 2 3; do                                                     
  D=/sys/class/drm/card$c/device                                         
  echo manual | sudo tee $D/power_dpm_force_performance_level > /dev/null
  echo $L     | sudo tee $D/pp_dpm_sclk > /dev/null                      
  echo "card$c: $(grep '\*' $D/pp_dpm_sclk)"                             
done                                                                     
