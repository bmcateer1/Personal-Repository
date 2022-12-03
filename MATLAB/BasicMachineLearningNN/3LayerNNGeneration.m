

X = [0 0 1;
 0 1 1;
 1 0 1;
 1 1 1;];
% x1 x2 x3
D = [0 0 1 1]'; %y
%NN
%Layer configuration
Layers = [
 featureInputLayer(3) %input layer

 fullyConnectedLayer(1)%hidden layer arbitrary setting to increase accuracy at cost of speed
 %had to keep nodes low otherwise it was TOO good and I couldnt show the
 %plot
 batchNormalizationLayer
 swishLayer

 fullyConnectedLayer(1) % output layer
 regressionLayer
 ];
%Training option
Options = trainingOptions('sgdm','InitialLearnRate',0.9,'MaxEpochs',200,'Plots','training-progress');
%Training
Network = trainNetwork(X,D,Layers,Options);
%Predict
PredNN = predict(Network,X);
%Analyze the network
analyzeNetwork(Layers)

plot(PredNN, D, '.')
title('With Batch Learning')
xlabel('Predicted')
ylabel('True Output')