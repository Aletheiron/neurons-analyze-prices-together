import numpy as np
import random
import copy
from copy import deepcopy
from collections import Counter
import pandas as pd


#Some exogeneous parameters

#List of utility function situations in the system

UF_system_list=[]
UF_first=np.float64(-1000000)
UF_system_list.append(UF_first)

#MSE Function of Joy. Utility function can be anything in general

def mse_joy (y, y_pred):
    
    mse_j = -1*(np.mean((y - y_pred)**2))
    
    return mse_j

def mse_joy_absolute (y, y_pred):
    
    mse_j = -1*(np.abs(np.mean((y - y_pred))))
    
    return mse_j

#List of nodes

list_of_nodes=[]

#Dictionary of nodes, where coordinates tuple is a key and node instance is a value. We need that for quik indentification of the node 
#which dendrite of something else is belonging to

dict_of_nodes={}

#Dictionary for descrite multiplyers for in-synapse computations
alpha_state_dict={'1':1,'2':1.2,'3':1.4, '4':1.6,'5':2}

#Treshold for synapse alpha change
treshold_synapse=0.8

#List of neuron of the system

list_of_neurons=[]

#Filter for the neuron computation in soma
neuron_filter=0.01


#Node environment parameter. Can be a separate function
#k=0.0000003

#Class of synapses, where the most computations are done

class Synapse():
    
    def __init__(self):
        
        self.node=None #parental node
        self.coordinates=None #coordinates are the same with node
        self.parent_dendrite=None #parental dendrite
        self.alpha_state=1 #alpha state key for updating alpha-value 
        self.alpha=1 #default alpha value
        self.connection_established=False #initial connection is False
        self.obtained_signal=0 #obtained signal
        
        self.signal_obtained_list=[0.00] #list of signals receoved in history. We need that list for handle time dependancies.
        self.value_list=[0.00] #list of values in history. We need that list for handle time dependancies. Zero is the first value.
        
        self.k=0.15 #check some hypothesis for node environment parameter
        
        #self.k=0.5/(1+self.time)
        
        self.axon_connected=None #connected axon
        self.axon_connected_list=[None, None] #special list for axon connection tracking
        
        self.connection_force=1 #connection force
        self.synapse_value=0 #value of synapse for sending to the neuron
        
        
        self.frozen=False #parameter for strong undestructable connections
        
        self.axon_breaking_connection=None
        self.axon_breaking_connection_list=[] #list of axons with disrapted connections
        
        self.connection_effort_count=0
        self.current_slow_try=False
        self.time=0 #timestamp for signal management
        
    
    
    def signal_receiving(self):
        
        '''If axon has signal in the last time-stamp, our synapse receives value'''
        
        #Case of positive value from axon last time-stamp
        if self.connection_established==True:
            #print(self.connection_established)
            #print(f'axon connected: {self.axon_connected}')
            if self.time>=1:
                self.obtained_signal=self.axon_connected.axon_signal_dict[self.time-1]
            else:
                #self.obtained_signal=self.axon_connected.axon_signal_dict[0]
                self.obtained_signal=0.00
        else:
            self.obtained_signal=0.00  #When synapse doesn't have connection 
                    
        
        #Update signal list
        self.signal_obtained_list.append(self.obtained_signal)
        
        
    
    
    def alpha_state_method(self):
        
        ''' Method for wrangle alpha state for diverse computations in synapses'''
        
        #Case when synapse obtained some valuable signal
        if self.frozen==False:
            
            if np.abs(self.obtained_signal)>=treshold_synapse and self.axon_connected_list[-1]!=self.axon_connected_list[-2]:
                
                self.alpha_state=self.alpha_state+1
                
                #Situation of the highest state, we keep the highest possible state in the dictionnary
                if self.alpha_state>=5:
                    
                    self.alpha=alpha_state_dict['5']
                    
                else:
                    self.alpha=alpha_state_dict[str(self.alpha_state)]
            
            #The basic state for alpha    
            else:
                
                self.alpha_state=1
                self.alpha=alpha_state_dict[str(self.alpha_state)]
                
        #Basic state for frozen connections. Alpha state is also frozen
        else:
            
            self.alpha_state=1
            self.alpha=alpha_state_dict[str(self.alpha_state)]
        
        # Creation of the new synapse in case of preferreble growth
        if self.alpha_state>5 and UF_system_list[-1]>=UF_system_list[-2]:
            
            self.alpha_state=1
            self.alpha=alpha_state_dict[str(self.alpha_state)]
            
            new_internal_synapse_instance=Synapse()
            new_internal_synapse_instance.coordinates=self.coordinates
            new_internal_synapse_instance.parent_dendrite=self.parent_dendrite
            new_internal_synapse_instance.axon_connected=self.axon_connected
            new_internal_synapse_instance.connection_established=self.connection_established
            new_internal_synapse_instance.node=self.node
            new_internal_synapse_instance.time=self.time
            new_internal_synapse_instance.axon_connected_list.append(new_internal_synapse_instance.axon_connected)
            new_internal_synapse_instance.parent_dendrite.synapse_list.append(new_internal_synapse_instance)
            new_internal_synapse_instance.current_slow_try=True
    
            
    def forward(self):
        
        ''' Forward method for computations sending to parental neuron'''
        
        synapse_value=self.k*self.alpha*self.obtained_signal*self.connection_force
        
        #self.synapse_value=np.min((synapse_value, 1))
        
        self.synapse_value=synapse_value
        
        #Update value list
        self.value_list.append(self.synapse_value)
        
        return self.synapse_value
    
    
    def try_connection(self):
        
        ''' Try connection with axon available in the parental node'''
        
        if self.frozen==False:
            #Using random parameter for random implementing this function
            a=np.random.randn()
            if a>=0.00:
                
                if self.node.list_of_belonging_axons !=[]:
            
                    axon_try=random.choice(self.node.list_of_belonging_axons) 
                    
                    if axon_try!=None:
                        self.axon_connected=axon_try
                        self.connection_established=True
                        self.connection_force=1
                        
                        self.axon_connected_list.append(self.axon_connected)
                        self.connection_effort_count+=1
                    else:
                        self.axon_connected=axon_try
                        self.connection_established=False
                        self.connection_force=0
                        self.axon_connected_list.append(self.axon_connected)
                    
                    #print(f'local connection: {axon_try}')
                else:
                    axon_try=None
                    self.connection_established=False
                    self.connection_force=0
                    self.axon_connected_list.append(self.axon_connected)
                
     
    def slow_try_connection(self):
        
        #Also trying connectionm but with a slow rate
        if self.frozen==False:
            a=np.random.randn()
            if a>=2.50:
                
                if self.node.list_of_belonging_axons !=[]:
            
                    axon_try=random.choice(self.node.list_of_belonging_axons) 
                    
                    if axon_try!=None:
                        self.axon_connected=axon_try
                        self.connection_established=True
                        self.connection_force=1
                        
                        self.axon_connected_list.append(self.axon_connected)
                        self.current_slow_try=True
                        
                        self.connection_effort_count+=1
                    else:
                        self.axon_connected=axon_try
                        self.connection_established=False
                        self.connection_force=0
                        self.axon_connected_list.append(self.axon_connected)
                        self.current_slow_try=False
                    
                    #print(f'local connection: {axon_try}')
                else:
                    axon_try=None
                    self.connection_established=False
                    self.connection_force=0
                    self.axon_connected_list.append(self.axon_connected)
                    self.current_slow_try=False
    
    
    def try_connection_disruption(self):
        
        ''' Try to break existing connection with some probability'''
        
        
        if self.connection_established==True and self.frozen==False:
            
            b=np.random.randn()
            if b>=2.50:
                  
                self.axon_breaking_connection=self.axon_connected
                self.axon_breaking_connection_list.append(self.axon_connected)
                
                self.axon_connected=None
                self.connection_force=0
                self.axon_connected_list.append(self.axon_connected)
                self.connection_established=False
                #print(f'connection is disrupted with: {self.axon_breaking_connection}')
                #print(f'and synapse is connected with: {self.axon_connected} with status: {self.connection_established}')
            
           
    
    def disruption_tracking(self):
        
        '''Tracking influence of the disruption made'''
        
        #Here we compare utility function after breaking connection and before. In case of lowering we recunstruct this connection
        
        if self.axon_breaking_connection!=None:
            
            if UF_system_list[-1]<UF_system_list[-2]: #Maybe use strict < for more fast disruption
                
                self.connection_established=True
                self.axon_connected=self.axon_breaking_connection_list[-1]
                
                #print(f'renewed axon connected: {self.axon_connected}')
                self.connection_force=1
                self.axon_breaking_connection=None
                self.axon_connected_list.append(self.axon_connected)
                
            else:
                self.axon_breaking_connection=None
    
            
    def synapse_sleep(self):
        
        #If synapse can't find good connection long time, it is going to sleep
        
        if self.connection_effort_count>=5 and self.connection_established==False:
            
            self.connection_established=False
            
            self.frozen=True
            
            self.axon_connected=None
    
         
    def connection_tracking(self):
        
        '''
        Growing force of connections in case of growng the whole utility function. Better implement dendrite's own utility function.
        But now we have not additional circumstances for doing that
        '''
        #Need more work
        if self.connection_established==True and self.axon_connected!=None and self.current_slow_try==True:
            
            if self.frozen==False:
                
                if UF_system_list[-1]<=UF_system_list[-2]:
                    
                    self.connection_force=0
                    self.connection_established=False
                    self.axon_connected=None
                    self.axon_connected_list.append(self.axon_connected)
                    self.current_slow_try=False
                    
                else:
                    self.current_slow_try=False
                    
                
                  


class Dendrite():
    
    def __init__(self):
        
        self.end_coordinates=(np.random.randint(0,8), np.random.randint(0,8)) #coordinates of the dendrite't root end
        self.synapse_list=[] #synapses of the particular dendrite
        self.node_belonging_list=[] #list of nodes where dendrite body is laying
        self.dendrite_signal=0 #dendritic signal
        self.parent_neuron=None #parental neuron for revealing node where denrite is laying
    
        
    def dendrite_proceed(self):
        
        ''' Summing up all signals from synapses in the current time-stamp'''
        
        self.dendrite_signal=0
        for synapse in self.synapse_list:
            
            self.dendrite_signal+=synapse.synapse_value
        
        
        return self.dendrite_signal
    
    
    def moving(self):
        
        '''Dendrite searches the most attractive direction and makes a move'''
        
        list_of_attraction=[]
        
        list_of_gravity=[]
        
        #compute distances for each node
        for i in range(len(list_of_nodes)):
            
            #Technical list
            list_of_quasi_dist=[]
            
            #Compute for every coordinate in coordinates' tuple
            for j in range(len(self.end_coordinates)):
                dist=(self.end_coordinates[j]-list_of_nodes[i].coordinates[j])
                
                #print(dist)
                
                #Square of coordinates difference for euclidian distance
                distance=dist*dist
                list_of_quasi_dist.append(distance)
            
            #Not real but square of distance. But it doesn't really matter   
            big_distance=sum(list_of_quasi_dist)
            
            #print(big_distance)
            
            #Conditional distance taking into accounts energy of the node
            gravity_distance=list_of_nodes[i].node_energy**2/(big_distance+0.00001)
            
            #print(f' node energy: {list_of_nodes[i].node_energy}')
            
            #Cosntrcuting list of attraction for every node
            list_of_attraction.append(gravity_distance)
            
            #print(list_of_attraction)
            
        #compute quasi-probabilities from inverses of distances
        for k in range(len(list_of_attraction)):
            
            #eliminating 100% probability of beign in the current node 
        
            
            if (list_of_attraction[k]==0) and list_of_nodes[k].coordinates==self.end_coordinates: 
                gravity=0
            
            else:
                gravity=list_of_attraction[k]
            
            #Constructing gravity force for every node from node list   
            all_gravity_force=sum(list_of_attraction)
            true_gravity=gravity/all_gravity_force
            list_of_gravity.append(true_gravity)
        
        #Define target node, mean with the highest gravity
        target_node_index=np.argmax(list_of_gravity)
        target_coord=list_of_nodes[target_node_index]
        # print(f'target node index: {target_node_index} and node: {target_coord}')
        # print(f'list of axon in the chosen node: {target_coord.list_of_belonging_axons}')
        
        #Moving to the target node with some speed and updating coordinates of the dendrite
        #This time I use simple one square movement at once in direction of the desired node
        
        new_coord=[]
        
        for c in range(len(self.end_coordinates)):
            
            #Adding one point to coordinate c, if this part is lower that target one
            if self.end_coordinates[c]<target_coord.coordinates[c]:
                new_coord_local=self.end_coordinates[c]+1
            #We don't change coordinate because it equal to target   
            elif self.end_coordinates[c]==target_coord.coordinates[c]: 
                new_coord_local=self.end_coordinates[c]
            #Substract one point to coordinate c, if this part is lower that target one   
            else:
                new_coord_local=self.end_coordinates[c]-1
            #new_coord_local=self.end_coordinates[c]+b*((target_coord.coordinates[c])-self.end_coordinates[c]) #b is exogeneous speed factor of moving dendrites
            new_coord.append(new_coord_local)
        
        #Updating coordinate tuple
        self.end_coordinates=tuple(new_coord)
        #print(f'new coordinates of dendrite: {self.end_coordinates}')
    
    
    def initial_nodes_revealing(self):
        
        ''' This method is for counting nodes where dendrite is laying in the beggining'''
        
        #Save coordinate for do not touch original one
        end_coordinates=copy.deepcopy(self.end_coordinates)
        
        #Add node if end_coordinates == neuron.coordinates
        
        if end_coordinates==self.parent_neuron.coordinates:
            
            #we use coordinates like key in dictionnary of nodes to know node instance
            node_to_belong=dict_of_nodes[tuple(end_coordinates)]
            
            #append node to dendrite's list
            self.node_belonging_list.append(node_to_belong)
            
        #check equality of coordinatates in direction we count
        while end_coordinates!=self.parent_neuron.coordinates:
            
            #we use coordinates like key in dictionnary of nodes to know node instance
            node_to_belong=dict_of_nodes[tuple(end_coordinates)]
            
            #check for avoid double counting of nodes
            if node_to_belong not in self.node_belonging_list:
                
                self.node_belonging_list.append(node_to_belong)
            
            #same scheme for step by step moving toward neouron core coordinates   
            new_coord=[]
        
            for c in range(len(self.end_coordinates)):
                
                #Adding one point to coordinate c, if this part is lower that target one
                if end_coordinates[c]<self.parent_neuron.coordinates[c]:
                    
                    new_coord_local=end_coordinates[c]+1
                
                #We don't change coordinate because it equal to target   
                elif end_coordinates[c]==self.parent_neuron.coordinates[c]: 
                    
                    new_coord_local=end_coordinates[c]
                    
                #Substract one point to coordinate c, if this part is lower that target one   
                else:
                    
                    new_coord_local=end_coordinates[c]-1
                
                new_coord.append(new_coord_local)
                
            #Updating coordinate tuple
            end_coordinates=tuple(new_coord)
            #print(f'new coordinates of dendrite: {self.end_coordinates}')
    
    
    def acquire_node(self):
        
        '''in case of moving we check belonging to node'''
        
        #we use coordinates like key in dictionnary of nodes to know node instance
        node_to_belong=dict_of_nodes[tuple(self.end_coordinates)]
        
        #check for avoid double counting of nodes
        if node_to_belong not in self.node_belonging_list:
            
            self.node_belonging_list.append(node_to_belong)
    
    def create_synapse(self):
        
        ''''Create synapses in each node dendrite is belonging'''
        
        #Only for the last node
        node=self.node_belonging_list[-1]
        
        #Check situation if the last synapse has already existed in the last node
        if self.synapse_list[-1].node != node:
            
            #We create five node synapses in each node by particular dendrite, but it is voluntaty decision
            for i in range(5):
                
                new_synapse_instance=Synapse()
                new_synapse_instance.coordinates=node.coordinates
                new_synapse_instance.node=node
                new_synapse_instance.parent_dendrite=self
                self.synapse_list.append(new_synapse_instance)
    
    
    def create_initial_synapses(self):
        
        ''''Create synapses for the initial list of nodes'''
        
        for node in self.node_belonging_list:
            
            #We create five node synapses in each node by particular dendrite, but it is voluntaty decision
            for i in range(5):
                
                new_synapse_instance=Synapse()
                new_synapse_instance.coordinates=node.coordinates
                new_synapse_instance.node=node
                new_synapse_instance.parent_dendrite=self
                self.synapse_list.append(new_synapse_instance)



class Axon():
    
    '''
    Axon class for passing signals
    '''
    
    def __init__(self):
        
        self.coordinates=(np.random.randint(0,8), np.random.randint(low=0,high=8)) #coordinates of the neuron soma
        self.parent_neuron=None #parental neuron
        self.nodes_with_this_axon_list=[] #list with node where axon is laying
        self.axon_signal_list=[0.00] #list of signal. Need for time handle
        
        self.time=0 #timestamp
        self.axon_signal_dict={0:0.00} #signal dictionnary for true passing to synapses
    
    def signal_of_axon (self):
        
        #Signal of axon is the sum of dendrites'values passed through filter this time-stamp
        neuron_logits=self.parent_neuron.neuron_proceed()
        
        #Updat list of values
        self.axon_signal_list.append(neuron_logits)
        
        #Update dictionnary
        self.axon_signal_dict[self.time]=neuron_logits
        
        return neuron_logits
    
    def initial_nodes_revealing(self):
        
        ''' This method is for counting nodes where axon is laying at the begining'''
        
        #Save coordinate for do not touch original one
        end_coordinates=copy.deepcopy(self.coordinates)
        
        #Add node if end_coordinates == neuron.coordinates
        
        if end_coordinates==self.parent_neuron.coordinates:
            
            #we use coordinates like key in dictionnary of nodes to know node instance
            node_to_belong=dict_of_nodes[tuple(end_coordinates)]
            
            #append node to dendrite's list
            self.nodes_with_this_axon_list.append(node_to_belong)
        
        #check equality of coordinatates in direction we count
        while end_coordinates!=self.parent_neuron.coordinates:
            
            #we use coordinates like key in dictionnary of nodes to know node instance
            node_to_belong=dict_of_nodes[tuple(end_coordinates)]
            
            #check for avoid double counting of nodes
            if node_to_belong not in self.nodes_with_this_axon_list:
                
                self.nodes_with_this_axon_list.append(node_to_belong)
            
            #same scheme for step by step moving toward neouron core coordinates   
            new_coord=[]
        
            for c in range(len(self.coordinates)):
                
                #Adding one point to coordinate c, if this part is lower that target one
                if end_coordinates[c]<self.parent_neuron.coordinates[c]:
                    
                    new_coord_local=end_coordinates[c]+1
                
                #We don't change coordinate because it equal to target   
                elif end_coordinates[c]==self.parent_neuron.coordinates[c]: 
                    
                    new_coord_local=end_coordinates[c]
                    
                #Substract one point to coordinate c, if this part is lower that target one   
                else:
                    
                    new_coord_local=end_coordinates[c]-1
                
                new_coord.append(new_coord_local)
                
            #Updating coordinate tuple
            end_coordinates=tuple(new_coord)
            



class Neuron():
    
    '''
    Neuron sums all inputs and implement some filters. But this is huge field of researches
    '''
    
    def __init__(self):
        
        self.child_dendrites_list=[] #dendrite list of the particular neuron
        self.coordinates=(np.random.randint(0,8), np.random.randint(0,8)) #coordinates of the neuron soma
        self.child_axon_list=[] #axon list of the current neuron
    
    
    def create_dendrites(self):
        
        ''' Here we create initial dendrites'''
        
        for i in range(15): #15 is voluntary. Sufficient for now
            
            new_dendrite_instance=Dendrite()
            #Appoint neuron as parental neuron for the created dendrite
            new_dendrite_instance.parent_neuron=self
            self.child_dendrites_list.append(new_dendrite_instance) #update list of axons
            
    def create_axons(self):
    
        ''' Here we create initial axons'''
        
        for i in range(3):
            
            new_axon_instance=Axon()
            #Appoint neuron as parental neuron for the created axons
            new_axon_instance.parent_neuron=self
            self.child_axon_list.append(new_axon_instance) #update list of axons
        
    
    def neuron_proceed(self):
        
        '''
        Main computing function of the neuron
        '''
        
        all_together=0
        #Sum up all logits from child dendrites
        for dendrite in self.child_dendrites_list:
            dendrite_logits=dendrite.dendrite_proceed()
            #print(f'dendrite signal in neuron: {dendrite.obtained_signal}')
            #print(f'dendrite logits: {dendrite_logits}')
            all_together+=dendrite_logits
        
        #Passing through filter. Might be zero like in ReLU function. But useless now. 
        if np.abs(all_together)>=neuron_filter:
            
            neuron_logits=all_together
            
        else:
            neuron_logits=0.00
        
        if neuron_logits>=0:
            
            neuron_logits=np.min((neuron_logits, 1.00)) 
        
        else:
            
            neuron_logits=np.max((neuron_logits,-1.00))
            
        return neuron_logits


class Node():
    
    '''
    Nodes are for aggregation of the axons. Need for computing simplification in the very big models
    '''
    
    def __init__(self, coordinates):
        
        self.coordinates=coordinates #coordinates
        self.node_energy=1 #default energy
        self.list_of_belonging_axons=[None] #list of axons
        self.list_of_established_synapses=[] #established synapses
        
    
    def energy_count(self):
        
        ''' We count connected synapses in the current node taking into account connection force '''
        
        self.node_energy=1
        for synapse in self.list_of_established_synapses:
            self.node_energy+=1*synapse.connection_force
            
        
        return self.node_energy
    
    def axon_revealing(self):
        
        ''' Forming list of axons in the current node'''
        
        #Check each neuron
        for neuron in list_of_neurons:
            
            #Check every axon 
            for axon in neuron.child_axon_list:
                
                #Check if current node in the list of belonging nodes of axon
                if self in axon.nodes_with_this_axon_list:
                    
                    #Update list of axons in the particular node
                    self.list_of_belonging_axons.append(axon)
    
    def established_synapses_revealing(self):
        
        ''' Counting for established synapses '''
        
         #Check each neuron
        for neuron in list_of_neurons:
            
            #Check every dendrite 
            for dendrite in neuron.child_dendrites_list:
                
                #Check if current node in the list of belonging nodes of axon
                if self in dendrite.node_belonging_list:
                    
                    #Check every synapse belonging to dendrite
                    for synapse in dendrite.synapse_list:
                        
                        #Check condition for condition and coordinates hve coincedance
                        if synapse.connection_established == True and synapse.coordinates==self.coordinates:
                            
                            #Update list of synapse in the particular node
                            self.list_of_established_synapses.append(synapse)


#Speciall synapses for perception of the correspondent data

class Synapse_for_positive_returns(Synapse):
    
    #Here we pass some data to the current synapse
    def __init__(self):
        super().__init__()
        self.column_index=0
        self.data=[0]
        self.signal=self.data[self.column_index]
        self.frozen=True
        self.connection_established=True
        
        
    
    def signal_receiving(self):
        
        #Rewriting function for doing nothing
        pass
    
    def forward(self):
        
        #In that case it is inly obtained signal time connection force
        #Maybe it is better use some normilizing function, but it is not important now
        
        
        #Synapse takes only signals of log-returns higher then one
        if self.signal>=1.00:
            
            #For symmetry in comparison with postive signals
            self.obtained_signal=self.signal-1
        
        #And zero otherwise  
        else:
            
            self.obtained_signal=0.00
            
            
        self.synapse_value=self.obtained_signal*self.connection_force
        
        #Update value list
        self.value_list.append(self.synapse_value)
        
        return self.synapse_value

class Synapse_for_negative_returns(Synapse):
    
    #Here we pass some data to the current synapse
    def __init__(self):
        super().__init__()
        self.column_index=0
        self.data=[0]
        self.signal=self.data[self.column_index]
        self.frozen=True
        self.connection_established=True
        
        
        
    
    def signal_receiving(self):
        
        #Rewriting function for doing nothing
        pass
    
    def forward(self):
        
        #In that case it is inly obtained signal time connection force
        #Maybe it is better use some normilizing function, but it is not important now
        
        #Synapse takes only signals of log-returns higher then one
        if self.signal<1.00:
            
            #For symmetry in comparison with postive signals
            self.obtained_signal=1-self.signal
        
        #And zero otherwise  
        else:
            
            self.obtained_signal=0.00
        
        
        self.synapse_value=self.obtained_signal*self.connection_force
        
        #Update value list
        self.value_list.append(self.synapse_value)
        
        return self.synapse_value


class Synapse_for_calming_neuron(Synapse):
    
    #Case of inverting received signals
    
    def forward(self):
        
        
        #If we obtain signal we invert it
        synapse_value=(-1.00)*self.k*self.alpha*self.obtained_signal*self.connection_force
        
        #self.synapse_value=np.min((synapse_value, 1))
        
        self.synapse_value=synapse_value
        
        #Update value list
        self.value_list.append(self.synapse_value)
        
        
            
        return self.synapse_value


class Dendrite_for_calming_neuron(Dendrite):
    
    
    def create_synapse(self):
    
        ''''Create synapses in each node dendrite is belonging'''
        
        #Only for the last node
        node=self.node_belonging_list[-1]
        
        #Check situation if the last synapse has already existed in the last node
        if self.synapse_list[-1].node != node:
            
            #We create five node synapses in each node by particular dendrite, but it is voluntaty decision
            for i in range(5):
                
                new_synapse_instance=Synapse_for_calming_neuron()
                new_synapse_instance.coordinates=node.coordinates
                new_synapse_instance.node=node
                new_synapse_instance.parent_dendrite=self
                self.synapse_list.append(new_synapse_instance)
    
    def create_initial_synapses(self):
    
        ''''Create synapses for the initial list of nodes'''
        
        for node in self.node_belonging_list:
            
            #We create five node synapses in each node by particular dendrite, but it is voluntaty decision
            for i in range(5):
                
                new_synapse_instance=Synapse_for_calming_neuron()
                new_synapse_instance.coordinates=node.coordinates
                new_synapse_instance.node=node
                new_synapse_instance.parent_dendrite=self
                self.synapse_list.append(new_synapse_instance)

    
class Neuron_for_calming(Neuron):
    
    def create_dendrites(self):
        
        ''' Here we create initial dendrites'''
        
        for i in range(15): #15 is voluntary. Sufficient for now
            
            new_dendrite_instance=Dendrite_for_calming_neuron()
            #Appoint neuron as parental neuron for the created dendrite
            new_dendrite_instance.parent_neuron=self
            self.child_dendrites_list.append(new_dendrite_instance) #update list of axons
            
    def neuron_proceed(self):
        
        '''
        Main computing function of the neuron
        '''
        
        all_together=0
        #Sum up all logits from child dendrites
        for dendrite in self.child_dendrites_list:
            dendrite_logits=dendrite.dendrite_proceed()
            #print(f'dendrite signal in neuron: {dendrite.obtained_signal}')
            #print(f'dendrite logits: {dendrite_logits}')
            all_together+=dendrite_logits
        
       
        
        if np.abs(all_together)>=neuron_filter:
            
            neuron_logits=all_together
            
        else:
            neuron_logits=0.00
        
        if neuron_logits<=0:
            
            neuron_logits=np.max((neuron_logits,-1)) 
            
        else: 
            neuron_logits=np.min((neuron_logits, 1))
         
        return neuron_logits       
    
#Data


# data=np.array([
#             [1.2,1.3,1.5,1.8,1.9,1.3,1.45,1.125],
#             [1.1,1.2,0.8,1.35,0.95,1.25,0.74,1.034],
#             [1.2,1.5,0.7,1.2,0.5,1.8,0.76,1.3],
#             [1.5,0.78,1.2,0.95,1.1,0.88,0.34,0.9]
#              ])

# target=np.array([
#             [1.7],
#             [0.95],
#             [1.3],
#             [0.75]
#              ])

# data=np.array([
#             [1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1],
#             [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
#              ])

# target=np.array([
#             [1.8],
#             [0.2]
#              ])

limiter=0.8
all_data=pd.read_csv('nvda_data')
all_data=np.array(all_data)

data=all_data[:,:-1]
target=all_data[:,-1]
split=int(limiter*len(data))
#print(len(data), len(target))

train_data=data[-120:-100:]
train_target=target[-120:-100]
#print(len(train_data), len(train_target))

test_data=data[-100:-80,:]
test_target=target[-100:-80]   
                            #20 for 20 for 8 timestamps for 5 last counters: [0.33, 0.6, 0.45, 0.611, 0.222] 
                            #20 for 20 for 8 timestamps: [0.55, 0.65, 0.6111, 0.6875, ---  (NaN)]
                            #20 for 20 for 4 timestaps: [0.4, 0.4, 0.47, 0.55, 0.00] 
                            #10 for 10 for 4 timestamps: [0.4, 0.3, 0.5, 0.6, 0.4]
                            #40 for 40 8 timestamps: [0.525, 0.42]
                            # concistency for 8 timestamps 10 times for 20 periods [train_result, test_result]--[-120:-100, -100:-80]:
                            # [[0.6, 0.55], [0.65, 0.5], [0.6, 0.55],
                            # [0.4, 0.45], [0.35, 0.4], [0.75, 0.5888], [0.65, 0.6], [0.5, 0.5], [0.6666, 0.4444], [0.4, 0.45]]
                            # concistency for 8 timestamps 5 times for 40 periods to 40 periods [-140:-100,-100:-60]:
                            # [[0.5, 0.55], [0.2162, 0.4324], [0.5, 0.55], [0.4782, 0.3913], [0.475, 0.525]]
                            # concistency for 8 timestamps 10 times for 20 periods [train_result, test_result]--[-820:-800, -800:-780]:
                            # [[0.3, 0.55], [0.3, 0.45], [0.35, 0.3], [0.3, 0.55], [0.55, 0.5], 
                            # [0.65, 0.6], [0.7, 0.45], [0.5, 0.55], [0.7, 0.45], [0.45, 0.5]]
                            # concistency for 8 timestamps 5 times for 20 periods to 100 periods [-120:-100,-100:]:
                            # [[0.7333, 0.5555], [0.6, 0.5473], [0.5, 0.5232], [0.65, 0.5684], [0.65, 0.5348]]
#print(len(test_data), len(test_target))




# x=0
# y=0
# x=np.asarray((1.1,1.2,0.8,1.35,0.95,1.25,0.74,1.034),
#              (1.2,1.5,0.7,1.2,0.5,1.8,0.76,1.3))

# y=np.asarray((1.002),
#              (1.3))

#Node initialization and coordinates establishment
#Nodes have coordinates from 0 to 7
#Node creation and forming list of nodes and dictionary with coordinates as keys

for i in range(8):
    
    for j in range(8):
        
        node_instance=Node(coordinates=(i,j))
        #update list of nodes
        list_of_nodes.append(node_instance)
        #update dict of nodes key: coordinates, values: node instance
        dict_of_nodes[node_instance.coordinates]=node_instance


#Architecture Construction


#All neurons have five dendrites and three axons with all consequenced synapses. Except of perceptive neurons


#16 neurons with only one synapse and one dendrite. 8 for positive returns, 8 for negative
#Each has connection with data - neuron - and three axons in the closest nodes connected to paired P-N, P-P or N-N, 
#and neuron accumulated all signals

#Perceptive neurons list

perceptive_neurons_list=[]

#first synapse with connection with positive data
first_positive_perceptive_synapse=Synapse_for_positive_returns()

first_positive_perceptive_synapse.column_index=0

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_first_positive_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_first_positive_synapse.synapse_list.append(first_positive_perceptive_synapse)

#neuron initialization
neuron_with_first_positive_synapse=Neuron()
#child dendrite upgrade
neuron_with_first_positive_synapse.child_dendrites_list.append(dendrite_with_first_positive_synapse)

#axon initilization and setting initial coordinates

axon_1_from_neuron_with_positive_synapse=Axon()
axon_1_from_neuron_with_positive_synapse.parent_neuron=neuron_with_first_positive_synapse
axon_1_from_neuron_with_positive_synapse.coordinates=(0,0) #connected with P-N neuron
axon_1_from_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_first_positive_synapse.child_axon_list.append(axon_1_from_neuron_with_positive_synapse)

axon_2_from_neuron_with_positive_synapse=Axon()
axon_2_from_neuron_with_positive_synapse.parent_neuron=neuron_with_first_positive_synapse
axon_2_from_neuron_with_positive_synapse.coordinates=(1,0) #connected with P-P neuron
axon_2_from_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_first_positive_synapse.child_axon_list.append(axon_2_from_neuron_with_positive_synapse)

axon_3_from_neuron_with_positive_synapse=Axon()
axon_3_from_neuron_with_positive_synapse.parent_neuron=neuron_with_first_positive_synapse
axon_3_from_neuron_with_positive_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_first_positive_synapse.child_axon_list.append(axon_3_from_neuron_with_positive_synapse)

perceptive_neurons_list.append(neuron_with_first_positive_synapse)


#first synapse with connection with negative data
first_negative_perceptive_synapse=Synapse_for_negative_returns()
first_negative_perceptive_synapse.column_index=0

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_first_negative_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_first_negative_synapse.synapse_list.append(first_negative_perceptive_synapse)


#neuron initialization
neuron_with_first_negative_synapse=Neuron()

#child dendrite upgrade
neuron_with_first_negative_synapse.child_dendrites_list.append(dendrite_with_first_negative_synapse)

#axon initilization and setting initial coordinates

axon_1_from_neuron_with_negative_synapse=Axon()
axon_1_from_neuron_with_negative_synapse.parent_neuron=neuron_with_first_negative_synapse
axon_1_from_neuron_with_negative_synapse.coordinates=(0,0) #connected with P-N neuron
axon_1_from_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_first_negative_synapse.child_axon_list.append(axon_1_from_neuron_with_negative_synapse)

axon_2_from_neuron_with_negative_synapse=Axon()
axon_2_from_neuron_with_negative_synapse.parent_neuron=neuron_with_first_negative_synapse
axon_2_from_neuron_with_negative_synapse.coordinates=(1,1) #connected with N-N neuron
axon_2_from_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_first_negative_synapse.child_axon_list.append(axon_2_from_neuron_with_negative_synapse)

axon_3_from_neuron_with_negative_synapse=Axon()
axon_3_from_neuron_with_negative_synapse.parent_neuron=neuron_with_first_negative_synapse
axon_3_from_neuron_with_negative_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_first_negative_synapse.child_axon_list.append(axon_3_from_neuron_with_negative_synapse)

perceptive_neurons_list.append(neuron_with_first_negative_synapse)



#second synapse with connection with positive data
second_positive_perceptive_synapse=Synapse_for_positive_returns()
second_positive_perceptive_synapse.column_index=1

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_second_positive_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_second_positive_synapse.synapse_list.append(second_positive_perceptive_synapse)

#neuron initialization
neuron_with_second_positive_synapse=Neuron()
#child dendrite upgrade
neuron_with_second_positive_synapse.child_dendrites_list.append(dendrite_with_second_positive_synapse)

#axon initilization and setting initial coordinates

axon_1_from_2_neuron_with_positive_synapse=Axon()
axon_1_from_2_neuron_with_positive_synapse.parent_neuron=neuron_with_second_positive_synapse
axon_1_from_2_neuron_with_positive_synapse.coordinates=(0,1) #connected with P-N neuron
axon_1_from_2_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_second_positive_synapse.child_axon_list.append(axon_1_from_2_neuron_with_positive_synapse)

axon_2_from_2_neuron_with_positive_synapse=Axon()
axon_2_from_2_neuron_with_positive_synapse.parent_neuron=neuron_with_second_positive_synapse
axon_2_from_2_neuron_with_positive_synapse.coordinates=(1,0) #connected with P-P neuron
axon_2_from_2_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_second_positive_synapse.child_axon_list.append(axon_2_from_2_neuron_with_positive_synapse)

axon_3_from_2_neuron_with_positive_synapse=Axon()
axon_3_from_2_neuron_with_positive_synapse.parent_neuron=neuron_with_second_positive_synapse
axon_3_from_2_neuron_with_positive_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_2_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_second_positive_synapse.child_axon_list.append(axon_3_from_2_neuron_with_positive_synapse)

perceptive_neurons_list.append(neuron_with_second_positive_synapse)



#second synapse with connection with negative data
second_negative_perceptive_synapse=Synapse_for_negative_returns()
second_negative_perceptive_synapse.column_index=1

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_second_negative_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_second_negative_synapse.synapse_list.append(second_negative_perceptive_synapse)

#neuron initialization
neuron_with_second_negative_synapse=Neuron()
#child dendrite upgrade
neuron_with_second_negative_synapse.child_dendrites_list.append(dendrite_with_second_negative_synapse)

#axon initilization and setting initial coordinates

axon_1_from_2_neuron_with_negative_synapse=Axon()
axon_1_from_2_neuron_with_negative_synapse.parent_neuron=neuron_with_second_negative_synapse
axon_1_from_2_neuron_with_negative_synapse.coordinates=(0,1) #connected with P-N neuron
axon_1_from_2_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_second_negative_synapse.child_axon_list.append(axon_1_from_2_neuron_with_negative_synapse)

axon_2_from_2_neuron_with_negative_synapse=Axon()
axon_2_from_2_neuron_with_negative_synapse.parent_neuron=neuron_with_second_negative_synapse
axon_2_from_2_neuron_with_negative_synapse.coordinates=(1,1) #connected with N-N neuron
axon_2_from_2_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_second_negative_synapse.child_axon_list.append(axon_2_from_2_neuron_with_negative_synapse)

axon_3_from_2_neuron_with_negative_synapse=Axon()
axon_3_from_2_neuron_with_negative_synapse.parent_neuron=neuron_with_second_negative_synapse
axon_3_from_2_neuron_with_negative_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_2_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_second_negative_synapse.child_axon_list.append(axon_3_from_2_neuron_with_negative_synapse)

perceptive_neurons_list.append(neuron_with_second_negative_synapse)



#third synapse with connection with positive data
third_positive_perceptive_synapse=Synapse_for_positive_returns()
third_positive_perceptive_synapse.column_index=2

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_third_positive_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_third_positive_synapse.synapse_list.append(third_positive_perceptive_synapse)

#neuron initialization
neuron_with_third_positive_synapse=Neuron()
#child dendrite upgrade
neuron_with_third_positive_synapse.child_dendrites_list.append(dendrite_with_third_positive_synapse)

#axon initilization and setting initial coordinates

axon_1_from_3_neuron_with_positive_synapse=Axon()
axon_1_from_3_neuron_with_positive_synapse.parent_neuron=neuron_with_third_positive_synapse
axon_1_from_3_neuron_with_positive_synapse.coordinates=(0,2) #connected with P-N neuron
axon_1_from_3_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_third_positive_synapse.child_axon_list.append(axon_1_from_3_neuron_with_positive_synapse)

axon_2_from_3_neuron_with_positive_synapse=Axon()
axon_2_from_3_neuron_with_positive_synapse.parent_neuron=neuron_with_third_positive_synapse
axon_2_from_3_neuron_with_positive_synapse.coordinates=(1,2) #connected with P-P neuron
axon_2_from_3_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_third_positive_synapse.child_axon_list.append(axon_1_from_3_neuron_with_positive_synapse)

axon_3_from_3_neuron_with_positive_synapse=Axon()
axon_3_from_3_neuron_with_positive_synapse.parent_neuron=neuron_with_third_positive_synapse
axon_3_from_3_neuron_with_positive_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_3_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_third_positive_synapse.child_axon_list.append(axon_3_from_3_neuron_with_positive_synapse)


perceptive_neurons_list.append(neuron_with_third_positive_synapse)



#third synapse with connection with negative data
third_negative_perceptive_synapse=Synapse_for_negative_returns()
third_positive_perceptive_synapse.column_index=2

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_third_negative_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_third_negative_synapse.synapse_list.append(third_negative_perceptive_synapse)

#neuron initialization
neuron_with_third_negative_synapse=Neuron()
#child dendrite upgrade
neuron_with_third_negative_synapse.child_dendrites_list.append(dendrite_with_third_negative_synapse)

#axon initilization and setting initial coordinates

axon_1_from_3_neuron_with_negative_synapse=Axon()
axon_1_from_3_neuron_with_negative_synapse.parent_neuron=neuron_with_third_negative_synapse
axon_1_from_3_neuron_with_negative_synapse.coordinates=(0,2) #connected with P-N neuron
axon_1_from_3_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_third_negative_synapse.child_axon_list.append(axon_1_from_3_neuron_with_negative_synapse)

axon_2_from_3_neuron_with_negative_synapse=Axon()
axon_2_from_3_neuron_with_negative_synapse.parent_neuron=neuron_with_third_negative_synapse
axon_2_from_3_neuron_with_negative_synapse.coordinates=(1,3) #connected with N-N neuron
axon_2_from_3_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_third_negative_synapse.child_axon_list.append(axon_2_from_3_neuron_with_negative_synapse)

axon_3_from_3_neuron_with_negative_synapse=Axon()
axon_3_from_3_neuron_with_negative_synapse.parent_neuron=neuron_with_third_negative_synapse
axon_3_from_3_neuron_with_negative_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_3_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_third_negative_synapse.child_axon_list.append(axon_3_from_3_neuron_with_negative_synapse)

perceptive_neurons_list.append(neuron_with_third_negative_synapse)


#forth synapse with connection with positive data
forth_positive_perceptive_synapse=Synapse_for_positive_returns()
forth_positive_perceptive_synapse.column_index=3

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_forth_positive_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_forth_positive_synapse.synapse_list.append(forth_positive_perceptive_synapse)

#neuron initialization
neuron_with_forth_positive_synapse=Neuron()
#child dendrite upgrade
neuron_with_forth_positive_synapse.child_dendrites_list.append(dendrite_with_forth_positive_synapse)

#axon initilization and setting initial coordinates

axon_1_from_4_neuron_with_positive_synapse=Axon()
axon_1_from_4_neuron_with_positive_synapse.parent_neuron=neuron_with_forth_positive_synapse
axon_1_from_4_neuron_with_positive_synapse.coordinates=(0,3) #connected with P-N neuron
axon_1_from_4_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_forth_positive_synapse.child_axon_list.append(axon_1_from_4_neuron_with_positive_synapse)

axon_2_from_4_neuron_with_positive_synapse=Axon()
axon_2_from_4_neuron_with_positive_synapse.parent_neuron=neuron_with_forth_positive_synapse
axon_2_from_4_neuron_with_positive_synapse.coordinates=(1,2) #connected with P-P neuron
axon_2_from_4_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_forth_positive_synapse.child_axon_list.append(axon_2_from_4_neuron_with_positive_synapse)

axon_3_from_4_neuron_with_positive_synapse=Axon()
axon_3_from_4_neuron_with_positive_synapse.parent_neuron=neuron_with_forth_positive_synapse
axon_3_from_4_neuron_with_positive_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_4_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_forth_positive_synapse.child_axon_list.append(axon_3_from_4_neuron_with_positive_synapse)

perceptive_neurons_list.append(neuron_with_forth_positive_synapse)


#forth synapse with connection with negative data
forth_negative_perceptive_synapse=Synapse_for_negative_returns()
forth_negative_perceptive_synapse.column_index=3

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_forth_negative_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_forth_negative_synapse.synapse_list.append(forth_negative_perceptive_synapse)

#neuron initialization
neuron_with_forth_negative_synapse=Neuron()
#child dendrite upgrade
neuron_with_forth_negative_synapse.child_dendrites_list.append(dendrite_with_forth_negative_synapse)

#axon initilization and setting initial coordinates

axon_1_from_4_neuron_with_negative_synapse=Axon()
axon_1_from_4_neuron_with_negative_synapse.parent_neuron=neuron_with_forth_negative_synapse
axon_1_from_4_neuron_with_negative_synapse.coordinates=(0,3) #connected with P-N neuron
axon_1_from_4_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_forth_negative_synapse.child_axon_list.append(axon_1_from_4_neuron_with_negative_synapse)

axon_2_from_4_neuron_with_negative_synapse=Axon()
axon_2_from_4_neuron_with_negative_synapse.parent_neuron=neuron_with_forth_negative_synapse
axon_2_from_4_neuron_with_negative_synapse.coordinates=(1,3) #connected with N-N neuron
axon_2_from_4_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_forth_negative_synapse.child_axon_list.append(axon_1_from_4_neuron_with_negative_synapse)

axon_3_from_4_neuron_with_negative_synapse=Axon()
axon_3_from_4_neuron_with_negative_synapse.parent_neuron=neuron_with_forth_negative_synapse
axon_3_from_4_neuron_with_negative_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_4_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_forth_negative_synapse.child_axon_list.append(axon_3_from_4_neuron_with_negative_synapse)


perceptive_neurons_list.append(neuron_with_forth_negative_synapse)


#fifth synapse with connection with positive data
fifth_positive_perceptive_synapse=Synapse_for_positive_returns()
fifth_positive_perceptive_synapse.column_index=4

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_fifth_positive_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_fifth_positive_synapse.synapse_list.append(fifth_positive_perceptive_synapse)

#neuron initialization
neuron_with_fifth_positive_synapse=Neuron()
#child dendrite upgrade
neuron_with_fifth_positive_synapse.child_dendrites_list.append(dendrite_with_fifth_positive_synapse)

#axon initilization and setting initial coordinates

axon_1_from_5_neuron_with_positive_synapse=Axon()
axon_1_from_5_neuron_with_positive_synapse.parent_neuron=neuron_with_fifth_positive_synapse
axon_1_from_5_neuron_with_positive_synapse.coordinates=(0,4) #connected with P-N neuron
axon_1_from_5_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_fifth_positive_synapse.child_axon_list.append(axon_1_from_5_neuron_with_positive_synapse)

axon_2_from_5_neuron_with_positive_synapse=Axon()
axon_2_from_5_neuron_with_positive_synapse.parent_neuron=neuron_with_fifth_positive_synapse
axon_2_from_5_neuron_with_positive_synapse.coordinates=(1,4) #connected with P-P neuron
axon_2_from_5_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_fifth_positive_synapse.child_axon_list.append(axon_2_from_5_neuron_with_positive_synapse)

axon_3_from_5_neuron_with_positive_synapse=Axon()
axon_3_from_5_neuron_with_positive_synapse.parent_neuron=neuron_with_fifth_positive_synapse
axon_3_from_5_neuron_with_positive_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_5_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_fifth_positive_synapse.child_axon_list.append(axon_3_from_5_neuron_with_positive_synapse)


perceptive_neurons_list.append(neuron_with_fifth_positive_synapse)


#fifth synapse with connection with negative data
fifth_negative_perceptive_synapse=Synapse_for_negative_returns()
fifth_negative_perceptive_synapse.column_index=4

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_fifth_negative_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_forth_negative_synapse.synapse_list.append(fifth_negative_perceptive_synapse)

#neuron initialization
neuron_with_fifth_negative_synapse=Neuron()
#child dendrite upgrade
neuron_with_fifth_negative_synapse.child_dendrites_list.append(dendrite_with_fifth_negative_synapse)

#axon initilization and setting initial coordinates

axon_1_from_5_neuron_with_negative_synapse=Axon()
axon_1_from_5_neuron_with_negative_synapse.parent_neuron=neuron_with_fifth_negative_synapse
axon_1_from_5_neuron_with_negative_synapse.coordinates=(0,4) #connected with P-N neuron
axon_1_from_5_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_fifth_negative_synapse.child_axon_list.append(axon_1_from_5_neuron_with_negative_synapse)

axon_2_from_5_neuron_with_negative_synapse=Axon()
axon_2_from_5_neuron_with_negative_synapse.parent_neuron=neuron_with_fifth_negative_synapse
axon_2_from_5_neuron_with_negative_synapse.coordinates=(1,5) #connected with N-N neuron
axon_2_from_5_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_fifth_negative_synapse.child_axon_list.append(axon_2_from_5_neuron_with_negative_synapse)

axon_3_from_5_neuron_with_negative_synapse=Axon()
axon_3_from_5_neuron_with_negative_synapse.parent_neuron=neuron_with_fifth_negative_synapse
axon_3_from_5_neuron_with_negative_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_5_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_fifth_negative_synapse.child_axon_list.append(axon_3_from_5_neuron_with_negative_synapse)

perceptive_neurons_list.append(neuron_with_fifth_negative_synapse)



#sixth synapse with connection with positive data
sixth_positive_perceptive_synapse=Synapse_for_positive_returns()
sixth_positive_perceptive_synapse.column_index=5

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_sixth_positive_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_sixth_positive_synapse.synapse_list.append(sixth_positive_perceptive_synapse)

#neuron initialization
neuron_with_sixth_positive_synapse=Neuron()
#child dendrite upgrade
neuron_with_sixth_positive_synapse.child_dendrites_list.append(dendrite_with_sixth_positive_synapse)

#axon initilization and setting initial coordinates

axon_1_from_6_neuron_with_positive_synapse=Axon()
axon_1_from_6_neuron_with_positive_synapse.parent_neuron=neuron_with_sixth_positive_synapse
axon_1_from_6_neuron_with_positive_synapse.coordinates=(0,5) #connected with P-N neuron
axon_1_from_6_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_sixth_positive_synapse.child_axon_list.append(axon_1_from_6_neuron_with_positive_synapse)

axon_2_from_6_neuron_with_positive_synapse=Axon()
axon_2_from_6_neuron_with_positive_synapse.parent_neuron=neuron_with_sixth_positive_synapse
axon_2_from_6_neuron_with_positive_synapse.coordinates=(1,4) #connected with P-P neuron
axon_2_from_6_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_sixth_positive_synapse.child_axon_list.append(axon_2_from_6_neuron_with_positive_synapse)

axon_3_from_6_neuron_with_positive_synapse=Axon()
axon_3_from_6_neuron_with_positive_synapse.parent_neuron=neuron_with_sixth_positive_synapse
axon_3_from_6_neuron_with_positive_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_6_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_sixth_positive_synapse.child_axon_list.append(axon_3_from_6_neuron_with_positive_synapse)

perceptive_neurons_list.append(neuron_with_sixth_positive_synapse)



#sixth synapse with connection with negative data
sixth_negative_perceptive_synapse=Synapse_for_negative_returns()
sixth_negative_perceptive_synapse.column_index=5

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_sixth_negative_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_sixth_negative_synapse.synapse_list.append(sixth_negative_perceptive_synapse)

#neuron initialization
neuron_with_sixth_negative_synapse=Neuron()
#child dendrite upgrade
neuron_with_sixth_negative_synapse.child_dendrites_list.append(dendrite_with_sixth_negative_synapse)

#axon initilization and setting initial coordinates

axon_1_from_6_neuron_with_negative_synapse=Axon()
axon_1_from_6_neuron_with_negative_synapse.parent_neuron=neuron_with_sixth_negative_synapse
axon_1_from_6_neuron_with_negative_synapse.coordinates=(0,5) #connected with P-N neuron
axon_1_from_6_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_sixth_negative_synapse.child_axon_list.append(axon_1_from_6_neuron_with_negative_synapse)

axon_2_from_6_neuron_with_negative_synapse=Axon()
axon_2_from_6_neuron_with_negative_synapse.parent_neuron=neuron_with_sixth_negative_synapse
axon_2_from_6_neuron_with_negative_synapse.coordinates=(1,5) #connected with N-N neuron
axon_2_from_6_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_sixth_negative_synapse.child_axon_list.append(axon_2_from_6_neuron_with_negative_synapse)

axon_3_from_6_neuron_with_negative_synapse=Axon()
axon_3_from_6_neuron_with_negative_synapse.parent_neuron=neuron_with_sixth_negative_synapse
axon_3_from_6_neuron_with_negative_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_6_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_sixth_negative_synapse.child_axon_list.append(axon_3_from_6_neuron_with_negative_synapse)

perceptive_neurons_list.append(neuron_with_sixth_negative_synapse)



#seventh synapse with connection with positive data
seventh_positive_perceptive_synapse=Synapse_for_positive_returns()
seventh_positive_perceptive_synapse.column_index=6

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_seventh_positive_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_seventh_positive_synapse.synapse_list.append(seventh_positive_perceptive_synapse)

#neuron initialization
neuron_with_seventh_positive_synapse=Neuron()
#child dendrite upgrade
neuron_with_seventh_positive_synapse.child_dendrites_list.append(dendrite_with_seventh_positive_synapse)

#axon initilization and setting initial coordinates

axon_1_from_7_neuron_with_positive_synapse=Axon()
axon_1_from_7_neuron_with_positive_synapse.parent_neuron=neuron_with_seventh_positive_synapse
axon_1_from_7_neuron_with_positive_synapse.coordinates=(0,6) #connected with P-N neuron
axon_1_from_7_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_seventh_positive_synapse.child_axon_list.append(axon_1_from_7_neuron_with_positive_synapse)

axon_2_from_7_neuron_with_positive_synapse=Axon()
axon_2_from_7_neuron_with_positive_synapse.parent_neuron=neuron_with_seventh_positive_synapse
axon_2_from_7_neuron_with_positive_synapse.coordinates=(1,6) #connected with P-P neuron
axon_2_from_7_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_seventh_positive_synapse.child_axon_list.append(axon_2_from_7_neuron_with_positive_synapse)

axon_3_from_7_neuron_with_positive_synapse=Axon()
axon_3_from_7_neuron_with_positive_synapse.parent_neuron=neuron_with_seventh_positive_synapse
axon_3_from_7_neuron_with_positive_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_7_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_seventh_positive_synapse.child_axon_list.append(axon_3_from_7_neuron_with_positive_synapse)

perceptive_neurons_list.append(neuron_with_seventh_positive_synapse)



#seventh synapse with connection with negative data
seventh_negative_perceptive_synapse=Synapse_for_negative_returns()
seventh_negative_perceptive_synapse.column_index=6

#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_seventh_negative_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_seventh_negative_synapse.synapse_list.append(seventh_negative_perceptive_synapse)

#neuron initialization
neuron_with_seventh_negative_synapse=Neuron()
#child dendrite upgrade
neuron_with_seventh_negative_synapse.child_dendrites_list.append(dendrite_with_seventh_negative_synapse)

#axon initilization and setting initial coordinates

axon_1_from_7_neuron_with_negative_synapse=Axon()
axon_1_from_7_neuron_with_negative_synapse.parent_neuron=neuron_with_seventh_negative_synapse
axon_1_from_7_neuron_with_negative_synapse.coordinates=(0,6) #connected with P-N neuron
axon_1_from_7_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_seventh_negative_synapse.child_axon_list.append(axon_1_from_7_neuron_with_negative_synapse)

axon_2_from_7_neuron_with_negative_synapse=Axon()
axon_2_from_7_neuron_with_negative_synapse.parent_neuron=neuron_with_seventh_negative_synapse
axon_2_from_7_neuron_with_negative_synapse.coordinates=(1,7) #connected with N-N neuron
axon_2_from_7_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_seventh_negative_synapse.child_axon_list.append(axon_2_from_7_neuron_with_negative_synapse)

axon_3_from_7_neuron_with_negative_synapse=Axon()
axon_3_from_7_neuron_with_negative_synapse.parent_neuron=neuron_with_seventh_negative_synapse
axon_3_from_7_neuron_with_negative_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_7_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_seventh_negative_synapse.child_axon_list.append(axon_3_from_7_neuron_with_negative_synapse)

perceptive_neurons_list.append(neuron_with_seventh_negative_synapse)



#eighth synapse with connection with positive data
eighth_positive_perceptive_synapse=Synapse_for_positive_returns()
eighth_positive_perceptive_synapse.column_index=7
#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_eighth_positive_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_eighth_positive_synapse.synapse_list.append(eighth_positive_perceptive_synapse)

#neuron initialization
neuron_with_eighth_positive_synapse=Neuron()
#child dendrite upgrade
neuron_with_eighth_positive_synapse.child_dendrites_list.append(dendrite_with_eighth_positive_synapse)

#axon initilization and setting initial coordinates

axon_1_from_8_neuron_with_positive_synapse=Axon()
axon_1_from_8_neuron_with_positive_synapse.parent_neuron=neuron_with_eighth_positive_synapse
axon_1_from_8_neuron_with_positive_synapse.coordinates=(0,7) #connected with P-N neuron
axon_1_from_8_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_eighth_positive_synapse.child_axon_list.append(axon_1_from_8_neuron_with_positive_synapse)

axon_2_from_8_neuron_with_positive_synapse=Axon()
axon_2_from_8_neuron_with_positive_synapse.parent_neuron=neuron_with_eighth_positive_synapse
axon_2_from_8_neuron_with_positive_synapse.coordinates=(1,6) #connected with P-P neuron
axon_2_from_8_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_eighth_positive_synapse.child_axon_list.append(axon_2_from_8_neuron_with_positive_synapse)

axon_3_from_8_neuron_with_positive_synapse=Axon()
axon_3_from_8_neuron_with_positive_synapse.parent_neuron=neuron_with_eighth_positive_synapse
axon_3_from_8_neuron_with_positive_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_8_neuron_with_positive_synapse.initial_nodes_revealing()
neuron_with_eighth_positive_synapse.child_axon_list.append(axon_3_from_8_neuron_with_positive_synapse)

perceptive_neurons_list.append(neuron_with_eighth_positive_synapse)



#eighthsynapse with connection with negative data
eighth_negative_perceptive_synapse=Synapse_for_negative_returns()
eighth_negative_perceptive_synapse.column_index=7
#only one dendrite. This dendrite shouldn't move during learning
dendrite_with_eighth_negative_synapse=Dendrite()
#forming list of synapses. It is only one this situation
dendrite_with_eighth_negative_synapse.synapse_list.append(eighth_negative_perceptive_synapse)

#neuron initialization
neuron_with_eighth_negative_synapse=Neuron()
#child dendrite upgrade
neuron_with_eighth_negative_synapse.child_dendrites_list.append(dendrite_with_eighth_negative_synapse)

#axon initilization and setting initial coordinates

axon_1_from_8_neuron_with_negative_synapse=Axon()
axon_1_from_8_neuron_with_negative_synapse.parent_neuron=neuron_with_eighth_negative_synapse
axon_1_from_8_neuron_with_negative_synapse.coordinates=(0,7) #connected with P-N neuron
axon_1_from_8_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_eighth_negative_synapse.child_axon_list.append(axon_1_from_8_neuron_with_negative_synapse)

axon_2_from_8_neuron_with_negative_synapse=Axon()
axon_2_from_8_neuron_with_negative_synapse.parent_neuron=neuron_with_eighth_negative_synapse
axon_2_from_8_neuron_with_negative_synapse.coordinates=(1,7) #connected with N-N neuron
axon_2_from_8_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_eighth_negative_synapse.child_axon_list.append(axon_2_from_8_neuron_with_negative_synapse)

axon_3_from_8_neuron_with_negative_synapse=Axon()
axon_3_from_8_neuron_with_negative_synapse.parent_neuron=neuron_with_eighth_negative_synapse
axon_3_from_8_neuron_with_negative_synapse.coordinates=(5,3) #connected with 'Absorbong all' neuron
axon_3_from_8_neuron_with_negative_synapse.initial_nodes_revealing()
neuron_with_eighth_negative_synapse.child_axon_list.append(axon_3_from_8_neuron_with_negative_synapse)

perceptive_neurons_list.append(neuron_with_eighth_negative_synapse)


# Special mini-method for perceptive neurons:
# Synapse recieves signal --> dendrite proceed only one signal --> neuron proceed only one dendrite --> signal to axons -->
# connected synapses from other neurons receive signal
# Dendrites don't move nor create additional synapses


#8 neurons with connections with positive and negative neurons : 1.P-N
#Standard neuron but two dendrites have special cordinates. And two synapses have special coordinates and connected and frozen
#Axons are standard

#loop for creating neuron instances
for i in range(8):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(0,i)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    special_synapse_1.coordinates=(0,i)
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    special_synapse_1.axon_connected=perceptive_neurons_list[i].child_axon_list[0]
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special sendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(0,i)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    special_synapse_2.coordinates=(0,i)
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_2.axon_connected=perceptive_neurons_list[i+1].child_axon_list[0]
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing() 
        ''' Something strange around here'''
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)



#4 neurons with connections with pair of positive neurons: 1.P-P
#Standard neuron but two dendrites have special cordinates. And two synapses have special coordinates and connected and frozen
#And two special axons with connections in nodes for (PP)-(NN) and (PP)-(PP)

#loop for creating neuron instances
for i in range(4):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(1,2*i)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    special_synapse_1.coordinates=(1,2*i)
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    special_synapse_1.axon_connected=perceptive_neurons_list[2*i].child_axon_list[1]
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special sendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(0,2*i)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    special_synapse_2.coordinates=(1,2*i)
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_2.axon_connected=perceptive_neurons_list[2*i+2].child_axon_list[1]
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Update coordinates for the two special axons
    
    #Connections to PP-NN layer
    new_neuron_instance.child_axon_list[0].coordinates=(2,2+i)
    
    #Connections to PP-PP layer
    if i==0 or i==1:
        new_neuron_instance.child_axon_list[1].coordinates=(3,2)
    else:
        new_neuron_instance.child_axon_list[1].coordinates=(3,4)
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)
    

#4 neurons with connections with pair of negative neurons: 1.N-N
#Standard neuron but two dendrites have special cordinates. And two synapses have special coordinates and connected and frozen
#And two special axons with connections in nodes for (PP)-(NN) and (NN)-(NN)

#loop for creating neuron instances
for i in range(4):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(1,2*i+1)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    special_synapse_1.coordinates=(1,2*i+1)
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    special_synapse_1.axon_connected=perceptive_neurons_list[2*i+1].child_axon_list[1]
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special sendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(1,2*i+1)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    special_synapse_2.coordinates=(1,2*i+1)
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_2.axon_connected=perceptive_neurons_list[2*i+3].child_axon_list[1]
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Update coordinates for the two special axons
    
    #Connections to PP-NN layer
    new_neuron_instance.child_axon_list[0].coordinates=(2,2+i)
    
    #Connections to NN-NN layer
    if i==0 or i==1:
        new_neuron_instance.child_axon_list[1].coordinates=(3,3)
    else:
        new_neuron_instance.child_axon_list[1].coordinates=(3,5)
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)
    
#4 neurons with connections with pair of negative and pair of positive neurons: 2.(PP)-(NN)
#Standard neuron but two dendrites have special cordinates. And two synapses have special coordinates and connected and frozen
#And one special axon with connections in nodes for ((PP)-(NN) - (NN)-(NN))


#loop for creating neuron instances
for i in range(4):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(2,2*i)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    special_synapse_1.coordinates=(2,2*i)
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    special_synapse_1.axon_connected=list_of_neurons[8+i].child_axon_list[0]#!!!!! hard to choose convinent axon
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special sendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(2,2*i+1)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    special_synapse_2.coordinates=(2,2*i+1)
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_2.axon_connected=list_of_neurons[12+i].child_axon_list[0]
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Update coordinates for the special axons
    
    #Connections to (PP-NN)-(PP-NN) layer
    new_neuron_instance.child_axon_list[0].coordinates=(4,3+i//2)
    
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)
    
#2 neurons with connections with two pairs of positive neurons: 2.(PP)-(PP)
#Standard neuron but two dendrites have special cordinates. And two synapses have special coordinates and connected and frozen
#And one special axon with connections in nodes for ((PP)-(PP) - (PP)-(PP))

#loop for creating neuron instances
for i in range(2):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    if i==0:
        new_neuron_instance.child_dendrites_list[0].end_coordinates=(3,2)
    else:
        new_neuron_instance.child_dendrites_list[0].end_coordinates=(3,4)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    if i==0:
        special_synapse_1.coordinates=(3,2)
    else:
        special_synapse_1.coordinates=(3,4)
        
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    if i==0:
        special_synapse_1.axon_connected=list_of_neurons[8].child_axon_list[1]#!!!!! hard to choose convinent axon
    else:
        special_synapse_1.axon_connected=list_of_neurons[10].child_axon_list[1]
        
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special dendrite and synapse in it
    if i==0:
        new_neuron_instance.child_dendrites_list[1].end_coordinates=(3,2)
    else:
        new_neuron_instance.child_dendrites_list[1].end_coordinates=(3,4)
        
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    if i==0:
        special_synapse_2.coordinates=(3,2)
    else:
        special_synapse_2.coordinates=(3,4)
        
    
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    
    if i==0:
        special_synapse_2.axon_connected=list_of_neurons[9].child_axon_list[1]#!!!!! hard to choose convinent axon
    else:
        special_synapse_2.axon_connected=list_of_neurons[11].child_axon_list[1]
    
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Update coordinates for the special axons
    
    #Connections to (PP-PP)-(PP-PP) layer
    new_neuron_instance.child_axon_list[0].coordinates=(4,2)
    
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)


#2 neurons with connections with two pairs of negative neurons: 2.(NN)-(NN)
#Standard neuron but two dendrites have special cordinates. And two synapses have special coordinates and connected and frozen
#And one special axon with connections in nodes for ((NN)-(NN) - (NN)-(NN))

#loop for creating neuron instances
for i in range(2):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    if i==0:
        new_neuron_instance.child_dendrites_list[0].end_coordinates=(3,3)
    else:
        new_neuron_instance.child_dendrites_list[0].end_coordinates=(3,5)
        
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    if i==0:
        special_synapse_1.coordinates=(3,3)
    else:
        special_synapse_1.coordinates=(3,5)
        
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    if i==0:
        special_synapse_1.axon_connected=list_of_neurons[12].child_axon_list[1]#!!!!! hard to choose convinent axon
    else:
        special_synapse_1.axon_connected=list_of_neurons[14].child_axon_list[1]
        
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special dendrite and synapse in it
    if i==0:
        new_neuron_instance.child_dendrites_list[1].end_coordinates=(3,3)
    else:
        new_neuron_instance.child_dendrites_list[1].end_coordinates=(3,5)
        
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    if i==0:
        special_synapse_2.coordinates=(3,3)
    else:
        special_synapse_2.coordinates=(3,5)
        
    
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    if i==0:
        special_synapse_2.axon_connected=list_of_neurons[13].child_axon_list[1]#!!!!! hard to choose convinent axon
    else:
        special_synapse_2.axon_connected=list_of_neurons[15].child_axon_list[1]
    
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Update coordinates for the special axons
    
    #Connections to (NN-NN)-(NN-NN) layer
    new_neuron_instance.child_axon_list[0].coordinates=(4,5)
    
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)


#2 neurons: 3.((PP)-(NN))-((PP)-(NN))
#Standard neuron but two dendrites have special cordinates. And two synapses have special coordinates and connected and frozen
#And one special two axons with connections in nodes for output neurons

#loop for creating neuron instances
for i in range(2):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(4,3+i//2)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    special_synapse_1.coordinates=(4,3+i//2)
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    special_synapse_1.axon_connected=list_of_neurons[16+2*i].child_axon_list[0]#!!!!! hard to choose convinent axon
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special sendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(4,3+i//2)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    special_synapse_2.coordinates=(4,3+i//2)
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_2.axon_connected=list_of_neurons[17+2*i].child_axon_list[0]
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Update coordinates for the special axons
    
    #Connections to positive output neuron
    new_neuron_instance.child_axon_list[0].coordinates=(7,3)
    
    #Connections to negative output neurons
    new_neuron_instance.child_axon_list[1].coordinates=(7,4)
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)

#1 neuron for all positive neurons: 3.((PP)-(PP))-((PP)-(PP)))
#Standard neuron but two dendrites have special cordinates. And two synapses have special coordinates and connected and frozen
#And one special two axons with connections in nodes for output neurons
#loop for creating neuron instances

for i in range(1):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(4,2)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    special_synapse_1.coordinates=(4,2)
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    special_synapse_1.axon_connected=list_of_neurons[20].child_axon_list[0]#!!!!! hard to choose convinent axon
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special sendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(4,2)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    special_synapse_2.coordinates=(4,2)
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_2.axon_connected=list_of_neurons[21].child_axon_list[0]
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Update coordinates for the special axons
    
    #Connections to positive output neuron
    new_neuron_instance.child_axon_list[0].coordinates=(7,3)
    
    #Connections to negative output neurons
    new_neuron_instance.child_axon_list[1].coordinates=(7,4)
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)


#1 neuron for all negative neurons: 3.(((NN)-(NN))-((NN)-(NN)))
#Standard neuron but two dendrites have special cordinates. And two synapses have special coordinates and connected and frozen
#And one special two axons with connections in nodes for output neurons

#loop for creating neuron instances
for i in range(1):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(4,5)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    special_synapse_1.coordinates=(4,5)
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    special_synapse_1.axon_connected=list_of_neurons[22].child_axon_list[0]#!!!!! hard to choose convinent axon
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special sendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(4,5)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    special_synapse_2.coordinates=(4,5)
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_2.axon_connected=list_of_neurons[23].child_axon_list[0]
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Update coordinates for the special axons
    
    #Connections to positive output neuron
    new_neuron_instance.child_axon_list[0].coordinates=(7,3)
    
    #Connections to negative output neurons
    new_neuron_instance.child_axon_list[1].coordinates=(7,4)
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)

#1 neuron for all neurons: 4.PNPNPNPNPNPNPNPN 
#Standard neuron but two dendrites have special cordinates. And 16 synapses have special coordinates and connected and frozen
#And one special two axons with connections in nodes for output neurons

#loop for creating neuron instances
for i in range(1):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(5,3)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    for j in range(8):
        
        special_synapse_1=Synapse()
        special_synapse_1.coordinates=(5,3)
        special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
        special_synapse_1.connection_established=True
       
        special_synapse_1.axon_connected=perceptive_neurons_list[j].child_axon_list[2]#!!!!! hard to choose convinent axon
        special_synapse_1.frozen=True
        new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special sendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(5,3)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    
    for k in range(8):
        
        special_synapse_2=Synapse()
        special_synapse_2.coordinates=(5,3)
        special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
        special_synapse_2.connection_established=True
        
        special_synapse_2.axon_connected=perceptive_neurons_list[k+8].child_axon_list[2]
        special_synapse_2.frozen=True
        new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Update coordinates for the special axons
    
    #Connections to positive output neuron
    new_neuron_instance.child_axon_list[0].coordinates=(7,3)
    
    #Connections to negative output neurons
    new_neuron_instance.child_axon_list[1].coordinates=(7,4)
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)
    
    
#30 free neurons
#Standard neuron without special dendrites and axons

#loop for creating neuron instances
for i in range(15):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Create axons
    new_neuron_instance.create_axons()
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)



#15 free calming neurons
#Standard calming neuron without special dendrites and axons

#loop for creating neuron instances
for i in range(15):
    
    #Neuron instance
    new_neuron_instance=Neuron_for_calming()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Create axons
    new_neuron_instance.create_axons()
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    for axon in new_neuron_instance.child_axon_list:
        
        axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)
    
#one neuron for positive output and one for negative output, each is connected with all neurons with frozen synapses from level 3 and 4
#Standard neuron. Special four dendrites with four special synapses. And only one axon without nodes revealing. Only for reading signals of the whole
#system

#Positive outcome
for i in range(1):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(7,3)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    special_synapse_1.coordinates=(7,3)
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    special_synapse_1.axon_connected=list_of_neurons[24].child_axon_list[0]#!!!!! hard to choose convinent axon
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special dendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(7,3)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    special_synapse_2.coordinates=(7,3)
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_2.axon_connected=list_of_neurons[25].child_axon_list[0]
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    #The third special dendrite and synapse in it
    new_neuron_instance.child_dendrites_list[2].end_coordinates=(7,3)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_3=Synapse()
    special_synapse_3.coordinates=(7,3)
    special_synapse_3.parent_dendrite=new_neuron_instance.child_dendrites_list[2]
    special_synapse_3.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_3.axon_connected=list_of_neurons[26].child_axon_list[0]
    special_synapse_3.frozen=True
    new_neuron_instance.child_dendrites_list[2].synapse_list.append(special_synapse_3)
    
    #The forth special dendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(7,3)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_4=Synapse()
    special_synapse_4.coordinates=(7,3)
    special_synapse_4.parent_dendrite=new_neuron_instance.child_dendrites_list[3]
    special_synapse_4.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_4.axon_connected=list_of_neurons[27].child_axon_list[0]
    special_synapse_4.frozen=True
    new_neuron_instance.child_dendrites_list[3].synapse_list.append(special_synapse_3)
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Stay only one axon
    new_neuron_instance.child_axon_list.pop()
    new_neuron_instance.child_axon_list.pop()
    
  
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    # for axon in new_neuron_instance.child_axon_list:
        
    #     axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)


#Negative outcome
for i in range(1):
    
    #Neuron instance
    new_neuron_instance=Neuron()
    
    #Create dendrites. Two of them need the special treatment
    new_neuron_instance.create_dendrites()
    
    #Update coordinates for the first two dendrites 
    new_neuron_instance.child_dendrites_list[0].end_coordinates=(7,4)
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_1=Synapse()
    special_synapse_1.coordinates=(7,4)
    special_synapse_1.parent_dendrite=new_neuron_instance.child_dendrites_list[0]
    special_synapse_1.connection_established=True
    #Connection with the first axon from the i-perceptive neuron (positive)
    special_synapse_1.axon_connected=list_of_neurons[24].child_axon_list[1]#!!!!! hard to choose convinent axon
    special_synapse_1.frozen=True
    new_neuron_instance.child_dendrites_list[0].synapse_list.append(special_synapse_1)
    
    #The second special dendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(7,4)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_2=Synapse()
    special_synapse_2.coordinates=(7,4)
    special_synapse_2.parent_dendrite=new_neuron_instance.child_dendrites_list[1]
    special_synapse_2.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_2.axon_connected=list_of_neurons[25].child_axon_list[1]
    special_synapse_2.frozen=True
    new_neuron_instance.child_dendrites_list[1].synapse_list.append(special_synapse_2)
    
    #The third special dendrite and synapse in it
    new_neuron_instance.child_dendrites_list[2].end_coordinates=(7,4)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_3=Synapse()
    special_synapse_3.coordinates=(7,4)
    special_synapse_3.parent_dendrite=new_neuron_instance.child_dendrites_list[2]
    special_synapse_3.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_3.axon_connected=list_of_neurons[26].child_axon_list[1]
    special_synapse_3.frozen=True
    new_neuron_instance.child_dendrites_list[2].synapse_list.append(special_synapse_3)
    
    #The forth special dendrite and synapse in it
    new_neuron_instance.child_dendrites_list[1].end_coordinates=(7,4)
    
    #Create special frozen synapses with corresponding coordinates and parameters
    special_synapse_4=Synapse()
    special_synapse_4.coordinates=(7,4)
    special_synapse_4.parent_dendrite=new_neuron_instance.child_dendrites_list[3]
    special_synapse_4.connection_established=True
    #Connection with the first axon from the i+1-perceptive neuron (negative)
    special_synapse_4.axon_connected=list_of_neurons[27].child_axon_list[1]
    special_synapse_4.frozen=True
    new_neuron_instance.child_dendrites_list[3].synapse_list.append(special_synapse_3)
    
    #Create axons
    new_neuron_instance.create_axons()
    
    #Stay only one axon
    new_neuron_instance.child_axon_list.pop()
    new_neuron_instance.child_axon_list.pop()
        
    #Create all other synapses and nodes revealing
    for dendrite in new_neuron_instance.child_dendrites_list:
        
        dendrite.initial_nodes_revealing()
        dendrite.create_initial_synapses()
    
    #for every axon we need nodes revealing
    # for axon in new_neuron_instance.child_axon_list:
        
    #     axon.initial_nodes_revealing()
    
    #Update global neuron list
    list_of_neurons.append(new_neuron_instance)

       
#Parameters for training

time_for_training=8 #How many timestamps we give to load for the same data. Here 10 loads by 1000 time-stamps

great_time_for_training=1

epoches=2

#Main Scheme for trainig      

#Doings before main loop

#Some list for analysing the most profitable time-stamp
list_of_optimum_place=[]

#Node axons revealing
for node in list_of_nodes:
    
    #Forming list of axons in the current node
    node.axon_revealing()

#Epoch means training on the whole train data
for epoch in range(epoches):
    
    #We need mechanism load of initial data

    for row in range(len(train_data)):
        
        #we load every point separatly and make training on it
        print(f'current row of data is: {row}')
        y=train_target[row]
        
        for neuron in perceptive_neurons_list:
                    
            for dendrite in neuron.child_dendrites_list:
                
                for synapse in dendrite.synapse_list:
                    
                    synapse.data=train_data[row,:]
                    synapse.signal=synapse.data[synapse.column_index]


        #For every line in data matrix we give 10 or 5 views, and all this scheme we iterate number of epochs

        for greate_time in range(great_time_for_training):
            
            
            #Refresh system for clean distribution of the signal
            #Structure is the same but information in the system is absent. From the beggining signal flows into network

            for neuron in perceptive_neurons_list:
                        
                for dendrite in neuron.child_dendrites_list:
                    
                    for synapse in dendrite.synapse_list:
                        
                        synapse.alpha_state=1 #alpha state key for updating alpha-value 
                        synapse.alpha=1 
                        synapse.synapse_value=0
                        synapse.signal_obtained_list=[0.00] #list of signals receoved in history. We need that list for handle time dependancies.
                        synapse.value_list=[0.00]
                        synapse.time=0
                        synapse.axon_breaking_connection=None
                        synapse.axon_breaking_connection_list=[]
                    
                for axon in neuron.child_axon_list:
                    
                    axon.axon_signal_list=[0.00] #list of signal. Need for time handle
                
                    axon.time=0 #timestamp
                    axon.axon_signal_dict={0:0.00} #signal dictionnary for true passing to synapses 
            
            for neuron in list_of_neurons:
            
                #Every dendrite in the system moves the most attractive node
                for dendrite in neuron.child_dendrites_list:
                    
                    for synapse in dendrite.synapse_list:
                        
                        synapse.alpha_state=1 #alpha state key for updating alpha-value 
                        synapse.alpha=1 
                        synapse.synapse_value=0
                        synapse.obtained_signal=0
                        synapse.signal_obtained_list=[0.00] #list of signals receoved in history. We need that list for handle time dependancies.
                        synapse.value_list=[0.00]
                        synapse.time=0
                        synapse.axon_breaking_connection=None
                        synapse.axon_breaking_connection_list=[]
                        
                for axon in neuron.child_axon_list:
                    
                    axon.axon_signal_list=[0.00] #list of signal. Need for time handle
                
                    axon.time=0 #timestamp
                    axon.axon_signal_dict={0:0.00} #signal dictionnary for true passing to synapses

            UF_current_run_list=[]

            #for every timestamp available
            for timestamp in range (time_for_training):
                
                
                #Do all we need for nodes for energy counting
                for node in list_of_nodes:
                    
                    #Forming list of established synapses with connection in the current node
                    node.established_synapses_revealing()
                    
                    #Counting energy for every node
                    node.energy_count()
                
                
                #We must destiguish perceptive neurons-dendrites. They don't move.
                if timestamp<=0:
                    
                    for neuron in perceptive_neurons_list:
                        
                        for dendrite in neuron.child_dendrites_list:
                            
                            for synapse in dendrite.synapse_list:
                                
                                #signal proceeding
                                synapse.forward()
                            
                            dendrite.dendrite_proceed()
                            
                        for axon in neuron.child_axon_list:
                            
                            axon.time=timestamp
                        
                            axon.signal_of_axon()
                else: 
                    
                    for neuron in perceptive_neurons_list:
                        
                        for axon in neuron.child_axon_list:
                            
                            axon.time=timestamp
                            #Zero signal for all axons originated from perceptive neurons
                            axon.axon_signal_list.append(0.00)
                            axon.axon_signal_dict[axon.time]=0.00
                
                
                
                
                
                #Dendrites are moving toward the most attractive ones
                
                #We need trying new connections for un-established synapses and proceeding information in already established ones
                #neuron_number=0
                
                for neuron in list_of_neurons:
                    
                    
                    
                    #Every dendrite in the system moves the most attractive node
                    for dendrite in neuron.child_dendrites_list:
                        
                    
                        #Dendrite moves
                        dendrite.moving()
                        
                        #Add node to the list of belonging if it is allowed
                        dendrite.acquire_node()
                        
                        #Creating new synapses if it is needed
                        
                        dendrite.create_synapse()
                        
                        #For un-established synapses we try connections
                        #synapse_number=0
                        for synapse in dendrite.synapse_list:
                            
                            synapse.time=timestamp
                            #synapse.k=0.1/(1+timestamp)
                            #synapse_number+=1
                            
                            if timestamp==0 and greate_time==0:
                                if synapse.connection_established==False:
                                    
                                    #synapse tries connection
                                    synapse.try_connection()
                                    
                            # Here we need special slow function of connection seeking
                            else:
                                if synapse.connection_established==False:
                                    synapse.slow_try_connection()
                            
                            synapse.try_connection_disruption()
                            
                            #signal receiving
                            # print(f'neuron number: {neuron_number} and dendrite number: {dendrite_number} and synapse number: {synapse_number}')
                            # print(f'probable axon coordinates: {list_of_neurons[9].child_axon_list[1].coordinates}')
                            # print(f'dendrite coordinates:{dendrite.end_coordinates}')
                            # print(f'synapse coordinates:{synapse.coordinates}')
                            synapse.signal_receiving()
                            
                            #alpha-state update
                            synapse.alpha_state_method()
                            
                            #signal proceeding
                            synapse.forward()
                            
                        #Dendrite proceeds signals from its synapses
                        
                        dendrite.dendrite_proceed()
                        
                    #Neuron sum up all information available and pass through filter
                    
                    #For axons we reveal signals
                    
                    for axon in neuron.child_axon_list:
                        
                        axon.time=timestamp
                        axon.signal_of_axon()
                
                
                
            
                #Now system computes Utility function
                 
                
                
                #It is simple sum of utility functions
                
                # print(f'signal of the positive neuron output: {list_of_neurons[-2].neuron_proceed()}')
                # print(f'signal of the negative neuron output: {list_of_neurons[-1].neuron_proceed()}')
                
              
                
                if y>=1:
                    
                    if list_of_neurons[-2].neuron_proceed()>list_of_neurons[-1].neuron_proceed():
                        
                        UF_system=1
                        
                    else:
                        
                        UF_system=-3
                else:
                    
                    if list_of_neurons[-2].neuron_proceed()<list_of_neurons[-1].neuron_proceed():
                        
                        UF_system=1
                        
                    else:
                        
                        UF_system=-3
                        
                        
                # if y>=1:
                    
                #     if list_of_neurons[-2].neuron_proceed()>list_of_neurons[-1].neuron_proceed():
                        
                #         UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                    
                #     else:
                        
                #         UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                
                # else:
                    
                #     if list_of_neurons[-2].neuron_proceed()<list_of_neurons[-1].neuron_proceed():
                        
                #         UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                        
                #     else:
                        
                #         UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                
                
                #UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                
                # if y>=1:
                    #UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())
                #else:
                    #UF_system=mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                    
                #UF_system=mse_joy_absolute(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())
                
                print(f'big time stamp is: {greate_time}')
                print(f'timestamp: {timestamp}')
                print(f'for data row: {row}')
                UF_current_run_list.append(UF_system)
                
                # place_of_optimum=np.argmax(UF_current_run_list)
                # optimum=np.max(UF_current_run_list)
                # print(f'place of optimum is: {place_of_optimum} and optimum is: {optimum}')
                
                #Update list of utility function
                UF_system_list.append(UF_system)
                #print(f'Utility list: {UF_system_list}')
                #print(f'Utility function: {UF_system}')
                
                
                
                #Every synapse should track its connections
                for neuron in list_of_neurons:
                    
                    for dendrite in neuron.child_dendrites_list:
                        
                        for synapse in dendrite.synapse_list:
                            
                            synapse.connection_tracking()
                            
                            synapse.disruption_tracking()
                            
                            synapse.synapse_sleep()
                
                        
                        
                       
                #print(f'signal of the positive neuron output: {list_of_neurons[-2].neuron_proceed()}')
                #print(f'signal of the negative neuron output: {list_of_neurons[-1].neuron_proceed()}')


            #Refresh system for clean distribution of the signal
            #Structure is the same but information in the system is absent. From the beggining signal flows into network

            for neuron in perceptive_neurons_list:
                        
                for dendrite in neuron.child_dendrites_list:
                    
                    for synapse in dendrite.synapse_list:
                        
                        synapse.alpha_state=1 #alpha state key for updating alpha-value 
                        synapse.alpha=1 
                        synapse.synapse_value=0
                        synapse.signal_obtained_list=[0.00] #list of signals receoved in history. We need that list for handle time dependancies.
                        synapse.value_list=[0.00]
                        synapse.time=0
                        synapse.axon_breaking_connection=None
                        synapse.axon_breaking_connection_list=[]
                    
                for axon in neuron.child_axon_list:
                    
                    axon.axon_signal_list=[0.00] #list of signal. Need for time handle
                
                    axon.time=0 #timestamp
                    axon.axon_signal_dict={0:0.00} #signal dictionnary for true passing to synapses 

            for neuron in list_of_neurons:
                
                #Every dendrite in the system moves the most attractive node
                for dendrite in neuron.child_dendrites_list:
                    
                    for synapse in dendrite.synapse_list:
                        
                        synapse.alpha_state=1 #alpha state key for updating alpha-value 
                        synapse.alpha=1 
                        synapse.synapse_value=0
                        synapse.obtained_signal=0
                        synapse.signal_obtained_list=[0.00] #list of signals receoved in history. We need that list for handle time dependancies.
                        synapse.value_list=[0.00]
                        synapse.time=0
                        synapse.axon_breaking_connection=None
                        synapse.axon_breaking_connection_list=[]
                        
                for axon in neuron.child_axon_list:
                    
                    axon.axon_signal_list=[0.00] #list of signal. Need for time handle
                
                    axon.time=0 #timestamp
                    axon.axon_signal_dict={0:0.00} #signal dictionnary for true passing to synapses
            
            place_of_optimum=np.argmax(UF_current_run_list)
            optimum=np.max(UF_current_run_list)
            print(f'place of optimum is: {place_of_optimum} and optimum is: {optimum}')
                

            UF_current_run_list_clean=[]

            #for every timestamp available
            for timestamp in range (time_for_training):
                
                #Do all we need for nodes for energy counting
                for node in list_of_nodes:
                    
                    #Forming list of established synapses with connection in the current node
                    node.established_synapses_revealing()
                    
                    #Counting energy for every node
                    node.energy_count()

                
                #We must destiguish perceptive neurons-dendrites. They don't move.
                if timestamp<=0:
                    
                    for neuron in perceptive_neurons_list:
                        
                        for dendrite in neuron.child_dendrites_list:
                            
                            for synapse in dendrite.synapse_list:
                                
                                #signal proceeding
                                synapse.forward()
                            
                            dendrite.dendrite_proceed()
                            
                        for axon in neuron.child_axon_list:
                            
                            axon.time=timestamp
                        
                            axon.signal_of_axon()
                else: 
                    
                    for neuron in perceptive_neurons_list:
                        
                        for axon in neuron.child_axon_list:
                            
                            axon.time=timestamp
                            #Zero signal for all axons originated from perceptive neurons
                            axon.axon_signal_list.append(0.00)
                            axon.axon_signal_dict[axon.time]=0.00
                
                
                for neuron in list_of_neurons:
                    
                    
                    #Every dendrite in the system moves the most attractive node
                    for dendrite in neuron.child_dendrites_list:
                        
                        for synapse in dendrite.synapse_list:
                            
                            synapse.time=timestamp
                            
                            #synapse.k=0.1/(1+timestamp)
                            
                            synapse.signal_receiving()
                            
                            #alpha-state update
                            synapse.alpha_state_method()
                            
                            #signal proceeding
                            synapse.forward()
                            
                        #Dendrite proceeds signals from its synapses
                        
                        dendrite.dendrite_proceed()
                        
                    #Neuron sum up all information available and pass through filter
                    
                    #For axons we reveal signals
                    
                    for axon in neuron.child_axon_list:
                        
                        axon.time=timestamp
                        axon.signal_of_axon()
                
                #Some ranking utility function for classification tasks
                
                if y>=1:
                    
                    if list_of_neurons[-2].neuron_proceed()>list_of_neurons[-1].neuron_proceed():
                        
                        UF_system=1
                        
                    else:
                        
                        UF_system=-3
                else:
                    
                    if list_of_neurons[-2].neuron_proceed()<list_of_neurons[-1].neuron_proceed():
                        
                        UF_system=1
                        
                    else:
                        
                        UF_system=-3
                        
                # if y>=1:
                    
                #     if list_of_neurons[-2].neuron_proceed()>list_of_neurons[-1].neuron_proceed():
                        
                #         UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                    
                #     else:
                        
                #         UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                
                # else:
                    
                #     if list_of_neurons[-2].neuron_proceed()<list_of_neurons[-1].neuron_proceed():
                        
                #         UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                        
                #     else:
                        
                #         UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())
                
                
                
                #UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())+mse_joy(y=y, y_pred=1-list_of_neurons[-1].neuron_proceed())   
                #UF_system=mse_joy(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())
                #UF_system=mse_joy_absolute(y=y, y_pred=1+list_of_neurons[-2].neuron_proceed())
                
                print(f'big time stamp is: {greate_time}')
                print(f'timestamp: {timestamp}') 
                
                print(f'for data row clean: {row}')
                #Update list of utility function
                UF_system_list.append(UF_system)
                UF_current_run_list_clean.append(UF_system)
                #print(f'Utility list: {UF_system_list}')
                
                #print(f'UF is: {UF_system}')
                # place_of_optimum=np.argmax(UF_current_run_list_clean)
                # optimum=np.max(UF_current_run_list_clean)
                
                
                #print(f'place of optimum is: {place_of_optimum} and optimum is: {optimum}')
                
                
                         
                print(f'signal of the positive neuron output: {list_of_neurons[-2].neuron_proceed()}')
                print(f'signal of the negative neuron output: {list_of_neurons[-1].neuron_proceed()}')
                
            place_of_optimum=np.argmax(UF_current_run_list_clean)
            optimum=np.max(UF_current_run_list_clean)
            print(f'place of optimum is: {place_of_optimum} and optimum is: {optimum}')
    
        if epoch==epoches-1:
            
            list_of_optimum_place.append(place_of_optimum)

print(f'list of optimal places: {list_of_optimum_place}')

#We find the most effective place of the timestamp after training, where the system receive the best answer the first time. But not the first and the sevond period
optimum_frequent=Counter(list_of_optimum_place).most_common(1)[0][0]


print(f'the final optimum place is: {optimum_frequent}')

if optimum_frequent ==0:
    
    optimum_frequent=Counter(list_of_optimum_place).most_common(2)[1][0]
    
    if optimum_frequent==1:
        
        optimum_frequent=Counter(list_of_optimum_place).most_common(3)[2][0]
        
print(f'the final optimum place is: {optimum_frequent}')

#Testing method

# test_data=np.array([
#             [1.2,1.3,1.4,1.5,1.4,1.12,1.8,1.1],
#             [0.3,0.9,0.5,0.4,0.2,0.7,0.5,0.8]
#              ])

# test_target=np.array([
#             [1.7],
#             [0.5]
#              ])

final_list=[]

for row in range(len(train_data)):
    
    #print(f'current row of data is: {row}')
    y=train_target[row]
    
    for neuron in perceptive_neurons_list:
                
        for dendrite in neuron.child_dendrites_list:
            
            for synapse in dendrite.synapse_list:
                
                synapse.data=test_data[row,:]
                synapse.signal=synapse.data[synapse.column_index]
                
                #print(f'synapse received signal is: {synapse.signal}')
    
    for neuron in perceptive_neurons_list:
                        
        for dendrite in neuron.child_dendrites_list:
            
            for synapse in dendrite.synapse_list:
                
                synapse.alpha_state=1 #alpha state key for updating alpha-value 
                synapse.alpha=1 
                synapse.synapse_value=0
                synapse.signal_obtained_list=[0.00] #list of signals receoved in history. We need that list for handle time dependancies.
                synapse.value_list=[0.00]
                synapse.time=0
                synapse.axon_breaking_connection=None
                synapse.axon_breaking_connection_list=[]
            
        for axon in neuron.child_axon_list:
            
            axon.axon_signal_list=[0.00] #list of signal. Need for time handle
        
            axon.time=0 #timestamp
            axon.axon_signal_dict={0:0.00} #signal dictionnary for true passing to synapses 
        
    for neuron in list_of_neurons:
    
        #Every dendrite in the system moves the most attractive node
        for dendrite in neuron.child_dendrites_list:
            
            for synapse in dendrite.synapse_list:
                
                synapse.alpha_state=1 #alpha state key for updating alpha-value 
                synapse.alpha=1 
                synapse.synapse_value=0
                synapse.obtained_signal=0
                synapse.signal_obtained_list=[0.00] #list of signals receoved in history. We need that list for handle time dependancies.
                synapse.value_list=[0.00]
                synapse.time=0
                synapse.axon_breaking_connection=None
                synapse.axon_breaking_connection_list=[]
                
        for axon in neuron.child_axon_list:
            
            axon.axon_signal_list=[0.00] #list of signal. Need for time handle
        
            axon.time=0 #timestamp
            axon.axon_signal_dict={0:0.00} #signal dictionnary for true passing to synapses

    UF_result_list=[]
    
    local_signal_up=[]
    local_signal_down=[]
    
    for timestamp in range (time_for_training):
                
        #Do all we need for nodes for energy counting
        for node in list_of_nodes:
            
            #Forming list of established synapses with connection in the current node
            node.established_synapses_revealing()
            
            #Counting energy for every node
            node.energy_count()

                
        #We must destiguish perceptive neurons-dendrites. They don't move.
        if timestamp<=0:
            
            for neuron in perceptive_neurons_list:
                
                for dendrite in neuron.child_dendrites_list:
                    
                    for synapse in dendrite.synapse_list:
                        
                        #signal proceeding
                        synapse.forward()
                        #print(f"synapse signal is: {synapse.forward()}")
                    
                    dendrite.dendrite_proceed()
                    
                for axon in neuron.child_axon_list:
                    
                    axon.time=timestamp
                
                    axon.signal_of_axon()
                    #print(f"axon signal is: {axon.signal_of_axon()}")
        else: 
            
            for neuron in perceptive_neurons_list:
                
                for axon in neuron.child_axon_list:
                    
                    axon.time=timestamp
                    #Zero signal for all axons originated from perceptive neurons
                    axon.axon_signal_list.append(0.00)
                    axon.axon_signal_dict[axon.time]=0.00
                    #print(f"axon signal is: {axon.signal_of_axon()}")
        
                
        for neuron in list_of_neurons:
            
            
            #Every dendrite in the system moves the most attractive node
            for dendrite in neuron.child_dendrites_list:
                
                for synapse in dendrite.synapse_list:
                    
                    synapse.time=timestamp
                    
                    #synapse.k=0.1/(1+timestamp)
                    
                    synapse.signal_receiving()
                    
                    #alpha-state update
                    synapse.alpha_state_method()
                    
                    #signal proceeding
                    synapse.forward()
                    
                #Dendrite proceeds signals from its synapses
                
                dendrite.dendrite_proceed()
                
            #Neuron sum up all information available and pass through filter
            
            #For axons we reveal signals
            
            for axon in neuron.child_axon_list:
                
                axon.time=timestamp
                axon.signal_of_axon()
        
        #Some ranking utility function for classification tasks
        
        if y>=1:
            
            if list_of_neurons[-2].neuron_proceed()>list_of_neurons[-1].neuron_proceed():
                
                UF_system=1
                
            else:
                
                UF_system=0
        else:
            
            if list_of_neurons[-2].neuron_proceed()<list_of_neurons[-1].neuron_proceed():
                
                UF_system=1
                
            else:
                
                UF_system=0

        print(f'timestamp: {timestamp}') 
                
        print(f'for data row clean: {row}')
        #Update list of utility function
        UF_result_list.append(UF_system)
        
        
        #print(f'for data: {test_data[row,:]}')
        print(f'signal of the positive neuron output: {list_of_neurons[-2].neuron_proceed()}')
        print(f'signal of the negative neuron output: {list_of_neurons[-1].neuron_proceed()}')
        
        local_signal_up.append(list_of_neurons[-2].neuron_proceed())
        local_signal_down.append(list_of_neurons[-1].neuron_proceed())
    
    
    result=UF_result_list[optimum_frequent]
    place_of_optimum=np.argmax(UF_result_list)
    optimum=np.max(UF_result_list)
    
    print(f'place of optimum is: {place_of_optimum} and optimum is: {optimum}')

    print(f'final list with result: {UF_result_list}')
    
    if local_signal_down[optimum_frequent]!=0.0 or local_signal_up[optimum_frequent]!=0.0:
        final_list.append(result)
    
    #final_list.extend(optimum)
    #final_list.append(result)


print(f'the most frequent place is: {optimum_frequent}')
accuracy=np.sum(final_list)/len(final_list)
print(f'final accuracy of the train data is: {accuracy}')


#Checking finding on the clean test data
final_list=[]

for row in range(len(test_data)):
    
    #print(f'current row of data is: {row}')
    y=test_target[row]
    
    for neuron in perceptive_neurons_list:
                
        for dendrite in neuron.child_dendrites_list:
            
            for synapse in dendrite.synapse_list:
                
                synapse.data=test_data[row,:]
                synapse.signal=synapse.data[synapse.column_index]
                
                #print(f'synapse received signal is: {synapse.signal}')
    
    for neuron in perceptive_neurons_list:
                        
        for dendrite in neuron.child_dendrites_list:
            
            for synapse in dendrite.synapse_list:
                
                synapse.alpha_state=1 #alpha state key for updating alpha-value 
                synapse.alpha=1 
                synapse.synapse_value=0
                synapse.signal_obtained_list=[0.00] #list of signals receoved in history. We need that list for handle time dependancies.
                synapse.value_list=[0.00]
                synapse.time=0
                synapse.axon_breaking_connection=None
                synapse.axon_breaking_connection_list=[]
            
        for axon in neuron.child_axon_list:
            
            axon.axon_signal_list=[0.00] #list of signal. Need for time handle
        
            axon.time=0 #timestamp
            axon.axon_signal_dict={0:0.00} #signal dictionnary for true passing to synapses 
        
    for neuron in list_of_neurons:
    
        #Every dendrite in the system moves the most attractive node
        for dendrite in neuron.child_dendrites_list:
            
            for synapse in dendrite.synapse_list:
                
                synapse.alpha_state=1 #alpha state key for updating alpha-value 
                synapse.alpha=1 
                synapse.synapse_value=0
                synapse.obtained_signal=0
                synapse.signal_obtained_list=[0.00] #list of signals receoved in history. We need that list for handle time dependancies.
                synapse.value_list=[0.00]
                synapse.time=0
                synapse.axon_breaking_connection=None
                synapse.axon_breaking_connection_list=[]
                
        for axon in neuron.child_axon_list:
            
            axon.axon_signal_list=[0.00] #list of signal. Need for time handle
        
            axon.time=0 #timestamp
            axon.axon_signal_dict={0:0.00} #signal dictionnary for true passing to synapses

    UF_result_list=[]
    
    local_signal_up=[]
    local_signal_down=[]
    
    for timestamp in range (time_for_training):
                
        #Do all we need for nodes for energy counting
        for node in list_of_nodes:
            
            #Forming list of established synapses with connection in the current node
            node.established_synapses_revealing()
            
            #Counting energy for every node
            node.energy_count()

                
        #We must destiguish perceptive neurons-dendrites. They don't move.
        if timestamp<=0:
            
            for neuron in perceptive_neurons_list:
                
                for dendrite in neuron.child_dendrites_list:
                    
                    for synapse in dendrite.synapse_list:
                        
                        #signal proceeding
                        synapse.forward()
                        #print(f"synapse signal is: {synapse.forward()}")
                    
                    dendrite.dendrite_proceed()
                    
                for axon in neuron.child_axon_list:
                    
                    axon.time=timestamp
                
                    axon.signal_of_axon()
                    #print(f"axon signal is: {axon.signal_of_axon()}")
        else: 
            
            for neuron in perceptive_neurons_list:
                
                for axon in neuron.child_axon_list:
                    
                    axon.time=timestamp
                    #Zero signal for all axons originated from perceptive neurons
                    axon.axon_signal_list.append(0.00)
                    axon.axon_signal_dict[axon.time]=0.00
                    #print(f"axon signal is: {axon.signal_of_axon()}")
        
                
        for neuron in list_of_neurons:
            
            
            #Every dendrite in the system moves the most attractive node
            for dendrite in neuron.child_dendrites_list:
                
                for synapse in dendrite.synapse_list:
                    
                    synapse.time=timestamp
                    
                    #synapse.k=0.1/(1+timestamp)
                    
                    synapse.signal_receiving()
                    
                    #alpha-state update
                    synapse.alpha_state_method()
                    
                    #signal proceeding
                    synapse.forward()
                    
                #Dendrite proceeds signals from its synapses
                
                dendrite.dendrite_proceed()
                
            #Neuron sum up all information available and pass through filter
            
            #For axons we reveal signals
            
            for axon in neuron.child_axon_list:
                
                axon.time=timestamp
                axon.signal_of_axon()
        
        #Some ranking utility function for classification tasks
        
        if y>=1:
            
            if list_of_neurons[-2].neuron_proceed()>list_of_neurons[-1].neuron_proceed():
                
                UF_system=1
                
            else:
                
                UF_system=0
        else:
            
            if list_of_neurons[-2].neuron_proceed()<list_of_neurons[-1].neuron_proceed():
                
                UF_system=1
                
            else:
                
                UF_system=0

        #print(f'timestamp: {timestamp}') 
                
        #print(f'for data row clean: {row}')
        #Update list of utility function
        UF_result_list.append(UF_system)
        
        
        #print(f'for data: {test_data[row,:]}')
        # print(f'signal of the positive neuron output: {list_of_neurons[-2].neuron_proceed()}')
        # print(f'signal of the negative neuron output: {list_of_neurons[-1].neuron_proceed()}')
        
        local_signal_up.append(list_of_neurons[-2].neuron_proceed())
        local_signal_down.append(list_of_neurons[-1].neuron_proceed())
    
    
    result=UF_result_list[optimum_frequent]
    place_of_optimum=np.argmax(UF_result_list)
    optimum=np.max(UF_result_list)
    
    print(f'place of optimum is: {place_of_optimum} and optimum is: {optimum}')

    print(f'final list with result: {UF_result_list}')
    
    if local_signal_down[optimum_frequent]!=0.0 or local_signal_up[optimum_frequent]!=0.0:
        final_list.append(result)
    
    #final_list.extend(optimum)
    #final_list.append(result)


print(f'the most frequent place is: {optimum_frequent}')
accuracy=np.sum(final_list)/len(final_list)
print(f'final accuracy is: {accuracy}')