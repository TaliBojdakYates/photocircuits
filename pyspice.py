import PySpice
import os
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# from PySpice.Spice.Simulation import Transient
# from PySpice.Spice.Simulation import dc

def solve(conectionDict,componentValuesInitial):
    biggestNode = 0
   
    circuit = Circuit('Resistor Bridge')
    gnd = circuit.gnd
    componentValues = []
    for inner_list in componentValuesInitial:
        digits = ""
        for char in inner_list[1]:
            if char.isdigit():
                digits += char
        componentValues.append([inner_list[0], digits, inner_list[2], inner_list[3]])



    # replace the biggest node number with 0
    for node in conectionDict:

        if biggestNode in conectionDict[node]:
            conectionDict[node][conectionDict[node].index(biggestNode)] = 0
   


    # Initialize value_to_key as an empty array
    connectionDict = {0: [0, 2], 1: [1, 2], 2: [0, 1]}

    # Create a new dictionary to store the connections between values and keys
    value_to_key = {}

    # Loop through the dictionary items
    for key, values in connectionDict.items():
        # Loop through the values of the current key
        for value in values:
            # If the value is already in the value_to_key dictionary, add the current key to its list of connections
            if value in value_to_key:
                value_to_key[value].append(key)
            # Otherwise, add the value to the value_to_key dictionary and set its connections to the current key
            else:
                value_to_key[value] = [key]

    # Print the connections between values and keys
    value_to_key_list = [[key, value_to_key[key]] for key in value_to_key]
   
    for i in value_to_key_list:
        for x in componentValues:
            if(i[0] == x[0]):
                x.append(i[1])
 
    for component in componentValues:
        if(component[4][0] == biggestNode):
            if component[3] == 0.0:
                
                circuit.V(component[0],component[4][1],gnd,component[1])
                
            elif component[3] == 1.0:
                
                circuit.R(component[0],component[4][1],gnd,component[1])
        elif(component[4][1] == biggestNode):
            if component[3] == 0.0:
                
                circuit.V(component[0],gnd,component[4][0],component[1])
                
            elif component[3] == 1.0:
                
                circuit.R(component[0],gnd,component[4][0],component[1])
        else:
            if component[3] == 0.0:
                
                circuit.V(component[0],component[4][0],component[4][1],component[1])
                
            elif component[3] == 1.0:
                
                circuit.R(component[0],component[4][0],component[4][1],component[1])
        # elif  component[3] == 2.0:
        #     print('dependent voltage source\n')
        # elif component[0] == 3.0:
        #     print('current source\n')
        #     val = input(f'what is the value for I{i}: ')
        # elif component[0] == 4.0:
        #     print('dependent current source\n')
        
      
   

    els = [i for i in circuit.element_names]
    for i in els:
        if 'R' in i:
           
            r = eval(f'circuit.{i}.plus.add_current_probe(circuit)')
        

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.operating_point()

    for node in analysis.branches.values():
        print('Node {}: {:5.2f} A'.format(str(node), float(node)))

    for node in analysis.nodes.values():
        print('Node {}: {:4.1f} V'.format(str(node), float(node)))
        
    
    
    
