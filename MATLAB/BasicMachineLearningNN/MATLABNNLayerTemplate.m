Data=newmatrix;
inputs = Data(:, 2:3);
output = Data(:, 1);
output2 = categorical(output);
%NN
%Layer configuration
Layers = [
 featureInputLayer(2) %input layer

 fullyConnectedLayer(10)%hidden layer arbitrary setting to increase accuracy at cost of speed
 %had to keep nodes low otherwise it was TOO good and I couldnt show the
 %plot
 batchNormalizationLayer
 %sigmoidlayer

 fullyConnectedLayer(20) % output layer
 batchNormalizationLayer
 %sigmoidlayer

 fullyConnectedLayer(2)
 softmaxLayer
 classificationLayer
];
%Training option
Options = trainingOptions('sgdm','InitialLearnRate',0.2,'MaxEpochs',500,'Plots','training-progress');
%Training
Network = trainNetwork(inputs,output2,Layers,Options);
%Predict
PredNN = predict(Network,inputs);
%Analyze the network
analyzeNetwork(Layers)

plot(PredNN, output2, '.')
title('With Batch Learning')
xlabel('Predicted')
ylabel('True Output')