Big mess in the code, I understand. But all findings are preliminary

I created a system of independently acting neurons that seek connections with each other based on their usefulness. And I tweaked it a little bit to predict price movements. Specifically, the direction of Nvidia stock prices. 
A little description. 
Synapse receives information by connection to axon. 
Then dendrite reunites signals from its synapses active this period. Each dendrite has many synapses. 
After that neuron sums up signals from its dendrites and passes it through filter. Then information flies into axons. And new time period begins. 
And now the most interesting part. Results.
I didn't train on all the available data for Nvidia stocks. I trained on just one run of twenty periods, and I checked the result on the next twenty periods. And I did this about twenty times. 
Here is the ranked result.
Let me remind you that we only predict the direction of movement. The main conclusion is this: if the accuracy during the training period is higher than 0.5 and lower than 0.66, then you can safely use the results of the intelligence hint. If the accuracy is higher than 0.66, overtraining is observed.
In fact, if the accuracy is below 0.5, you can use the network's predictions, but with the opposite sign. The result is not as stable, but almost always correct. 
Not bad, I think.
