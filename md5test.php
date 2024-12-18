<?php

function gettimems() { return floor(microtime(true) * 1000); }
  echo "Table is power of 2 byte input size, ms per hash, and throughput in MB/s.\n";
  $mystring = "12";
  $twopow = 1;
  $iterations = 100000;
  while(1) {
    $timestartms = gettimems();
    for($i=0; $i<$iterations; $i++) {
      $m = md5($mystring);
    }
    $timeendms = gettimems();
    $msperhash = ($timeendms - $timestartms)/$iterations;
    $kbpms = $iterations*(1 << $twopow)/(($timeendms - $timestartms)*1000.0);
    echo("$twopow $msperhash - $kbpms MB/s\n");
    $twopow+=1;
    $mystring .= $mystring;
  }

?>
