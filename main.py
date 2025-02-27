import machine 

import time 

import sys  

sensor_pir = machine.Pin(28, machine.Pin.IN) 

led = machine.Pin(25, machine.Pin.OUT) 

def pir_handler(pin): 

    print("ALARM! Motion detected!")   

    sys.stdout.write("MOTION_DETECTED\n")   

    led.value(1)   

    time.sleep(2)   

    led.value(0)   

sensor_pir.irq(trigger=machine.Pin.IRQ_RISING, handler=pir_handler) 

while True: 

    time.sleep(1)  
