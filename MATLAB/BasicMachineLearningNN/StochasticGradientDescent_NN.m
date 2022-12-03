clc
clear 
clear all %clears all previous data and plots

% XOR Truth table
% The number of input & output data
Xtrain(1,:) = [0 0];
Xtrain(2,:) = [0 1];
Xtrain(3,:) = [1 0];
Xtrain(4,:) = [1 1];
Ytrain = [0
 1
 1
 0];

Ytrain2 = categorical(Ytrain);
 % Layer configuration
SingleLayer = [
 featureInputLayer(2)

 fullyConnectedLayer(2)
 tanhLayer
 softmaxLayer
 classificationLayer

 ];

% Training option
Options = trainingOptions('sgdm','InitialLearnRate',0.2,'MaxEpochs',500,'Plots','training-progress');
% Training
Network = trainNetwork(Xtrain,Ytrain2,SingleLayer,Options);
% Testing
Xtest = Xtrain;
c_predicted = categorical(zeros(1,4));
for iii = 1:4
 c_predicted(iii) = classify(Network,Xtest(iii,:));
end
fprintf('\n x1 x2 y (predicted)\n');
for k = 1:4
 fprintf(' %d %d %c\n',Xtest(k,1),Xtest(k,2),c_predicted(k));
end
%Analyze the network
analyzeNetwork(SingleLayer)