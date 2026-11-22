F=/sys/bus/acpi/devices/APP0001:00                                                                        
for n in 1 2 3 4; do                                                                                      
  echo 1 | sudo tee $F/fan${n}_manual > /dev/null                                                         
  cat $F/fan${n}_max | sudo tee $F/fan${n}_output > /dev/null                                             
done                                                                                                      
sleep 10                                                                                                  
for n in 1 2 3 4; do echo "fan$n target=$(cat $F/fan${n}_output) actual=$(cat $F/fan${n}_input) rpm"; done
